"""ABOUTME: Weather MCP Server - Open-Meteo weather data with Nominatim geocoding.

Provides weather data via Open-Meteo API (no API key required). Uses OSM Nominatim for
geocoding to handle location names like "Savannah, GA". Returns dual-format responses:
human-readable text for LLM context + structured JSON for frontend rendering.
"""

import asyncio
import json
import logging
import os
from typing import Optional, Dict, Any, Literal, List
from datetime import datetime, timedelta
from enum import Enum

import httpx
from pydantic import BaseModel, Field, field_validator
from mcp.server.fastmcp import Context
from mcp.types import TextContent, CallToolResult

from common.mcp_base import MCPServerBase
from common.validation import validate_string_length, validate_non_empty_string
from common.error_handling import (
    ERROR_INVALID_INPUT,
    ERROR_TIMEOUT,
    ERROR_NOT_FOUND,
    ERROR_FETCH_FAILED,
    create_error_result,
    create_validation_error,
    create_network_error,
    create_timeout_error,
)
from common.http_utils import safe_http_get

# Initialize MCP server with base class
server = MCPServerBase("weather")
mcp = server.get_mcp()
logger = server.get_logger()

# ============================================================================
# CONSTANTS
# ============================================================================

# Geocoding and weather API endpoints
GEOCODING_API = "https://nominatim.openstreetmap.org/search"
WEATHER_API = "https://api.open-meteo.com/v1/forecast"

# Nominatim User-Agent (required by OSM policy)
USER_AGENT = "WeatherMCP/1.0 (https://github.com/anthropics/strieber-gpt-3)"

# API configuration
API_TIMEOUT_SECONDS = 10.0
GEOCODING_MAX_RESULTS = 1
GEOCODING_LANGUAGE = "en"

# Location validation
MAX_LOCATION_LENGTH = 256
MIN_LOCATION_LENGTH = 1

# Forecast limits
DAILY_FORECAST_HOURS = 24
WEEKLY_FORECAST_DAYS = 7
HOURLY_DISPLAY_INTERVAL = 3  # Show every 3rd hour in daily view

# Unit conversions
KMH_TO_MPH = 0.621371
CELSIUS_DISPLAY_UNIT = "°C"
FAHRENHEIT_DISPLAY_UNIT = "°F"

# API parameter specifications (as lists for clarity)
CURRENT_WEATHER_PARAMS = [
    "temperature_2m",
    "relative_humidity_2m",
    "weather_code",
    "wind_speed_10m",
    "apparent_temperature",
    "precipitation",
    "cloud_cover",
    "visibility",
    "uv_index",
    "pressure_msl",
    "dew_point_2m",
]

HOURLY_FORECAST_PARAMS = [
    "temperature_2m",
    "relative_humidity_2m",
    "weather_code",
    "wind_speed_10m",
    "precipitation",
    "precipitation_probability",
    "uv_index",
    "cloud_cover",
    "visibility",
    "dew_point_2m",
]

DAILY_FORECAST_PARAMS = [
    "temperature_2m_max",
    "temperature_2m_min",
    "weather_code",
    "wind_speed_10m_max",
    "precipitation_sum",
    "precipitation_probability_max",
    "uv_index_max",
    "sunrise",
    "sunset",
]

# Tool-specific error codes (beyond shared constants)
ERROR_CODE_INVALID_LOCATION = "invalid_location"
ERROR_CODE_LOCATION_NOT_FOUND = "location_not_found"
ERROR_CODE_AMBIGUOUS_LOCATION = "ambiguous_location"
ERROR_CODE_GEOCODE_SERVICE_ERROR = "geocode_service_error"
ERROR_CODE_INVALID_COORDINATES = "invalid_coordinates"
ERROR_CODE_INVALID_FORECAST_TYPE = "invalid_forecast_type"
ERROR_CODE_INVALID_UNITS = "invalid_units"

# Forecast type and unit enums
class ForecastType(str, Enum):
    """Valid forecast types."""
    CURRENT = "current"
    DAILY = "daily"
    WEEKLY = "weekly"

class UnitSystem(str, Enum):
    """Valid unit systems."""
    METRIC = "metric"
    IMPERIAL = "imperial"

# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class WeatherInputSchema(BaseModel):
    """Input schema for get_weather tool."""

    location: str = Field(
        ...,
        description="Location name (e.g., 'Paris', 'New York', 'Tokyo')",
        min_length=MIN_LOCATION_LENGTH,
        max_length=MAX_LOCATION_LENGTH
    )
    forecast_type: Literal["current", "daily", "weekly"] = Field(
        default="current",
        description="Type of forecast - 'current' for current weather, 'daily' for 24h forecast, 'weekly' for 7d forecast"
    )
    units: Literal["metric", "imperial"] = Field(
        default="imperial",
        description="Unit system - 'metric' (°C, km/h) or 'imperial' (°F, mph)"
    )

    @field_validator("location")
    @classmethod
    def validate_location(cls, v: str) -> str:
        """Validate and normalize location string.

        Args:
            v: Location string to validate

        Returns:
            Normalized location string

        Raises:
            ValueError: If location is invalid
        """
        # Strip whitespace
        v = v.strip()

        # Check non-empty
        if not v:
            raise ValueError("Location cannot be empty")

        # Check length
        if len(v) > MAX_LOCATION_LENGTH:
            raise ValueError(f"Location name too long (max {MAX_LOCATION_LENGTH} characters)")

        return v

class WeatherOutputSchema(BaseModel):
    """Output schema for get_weather tool."""

    location: str = Field(..., description="Resolved location name with country")
    latitude: float = Field(..., description="Latitude coordinate")
    longitude: float = Field(..., description="Longitude coordinate")
    forecast_type: str = Field(..., description="Type of forecast returned")
    units: str = Field(..., description="Temperature units used")
    timestamp: str = Field(..., description="ISO8601 timestamp of when data was fetched")
    data: Dict[str, Any] = Field(..., description="Structured weather data")

# ============================================================================
# WMO WEATHER CODES
# ============================================================================

# WMO Weather codes mapping
WMO_CODES: Dict[int, str] = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Foggy",
    48: "Foggy",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    71: "Slight snow",
    73: "Moderate snow",
    75: "Heavy snow",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}

# Emoji mappings for weather codes
EMOJI_CLEAR_SKY = "☀️"
EMOJI_PARTLY_CLOUDY = "⛅"
EMOJI_CLOUDY = "☁️"
EMOJI_FOG = "🌫️"
EMOJI_RAIN = "🌧️"
EMOJI_SNOW = "❄️"
EMOJI_THUNDERSTORM = "⛈️"
EMOJI_DEFAULT = "🌤️"


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def get_weather_emoji(code: int) -> str:
    """Get emoji for WMO weather code.

    Args:
        code: WMO weather code

    Returns:
        Emoji representing the weather condition
    """
    if code == 0:
        return EMOJI_CLEAR_SKY
    elif code in [1, 2]:
        return EMOJI_PARTLY_CLOUDY
    elif code == 3:
        return EMOJI_CLOUDY
    elif code in [45, 48]:
        return EMOJI_FOG
    elif code in [51, 53, 55, 80, 81, 82]:
        return EMOJI_RAIN
    elif code in [61, 63, 65]:
        return EMOJI_RAIN
    elif code in [71, 73, 75, 77, 85, 86]:
        return EMOJI_SNOW
    elif code in [95, 96, 99]:
        return EMOJI_THUNDERSTORM
    else:
        return EMOJI_DEFAULT

def get_unit_symbol(units: str) -> str:
    """Get temperature unit symbol.

    Args:
        units: Unit system ("metric" or "imperial")

    Returns:
        Unit symbol string
    """
    return FAHRENHEIT_DISPLAY_UNIT if units == "imperial" else CELSIUS_DISPLAY_UNIT

def get_wind_speed_unit(units: str) -> str:
    """Get wind speed unit based on unit system.

    Args:
        units: Unit system ("metric" or "imperial")

    Returns:
        Wind speed unit string ("mph" or "km/h")
    """
    return "mph" if units == "imperial" else "km/h"

# ============================================================================
# GEOCODING FUNCTIONS
# ============================================================================


async def geocode_location(location: str) -> Dict[str, Any]:
    """Convert location name to coordinates using OSM Nominatim Geocoding API.

    Nominatim handles all query formats reliably (e.g., "Savannah, GA", "Paris", etc).
    Requires User-Agent header per OSM usage policy.

    Args:
        location: Location name (e.g., "Paris", "New York", "Savannah, GA")

    Returns:
        Dict with 'name', 'latitude', 'longitude', 'country', 'timezone'

    Raises:
        ValueError: If location not found or invalid
        Exception: If API request fails
    """
    logger.debug(f"Geocoding location: '{location}'")

    try:
        params = {
            "q": location,
            "format": "json",
            "limit": GEOCODING_MAX_RESULTS,
            "addressdetails": 0
        }
        headers = {
            "User-Agent": USER_AGENT
        }
        resp = await safe_http_get(
            GEOCODING_API,
            params=params,
            headers=headers,
            timeout=API_TIMEOUT_SECONDS
        )

        data = resp.json()

        # Nominatim returns results array directly (not wrapped in "results" key)
        if not data:
            logger.warning(f"Location not found: '{location}'")
            raise ValueError(
                f"Location not found: {location}",
                {"error_code": ERROR_CODE_LOCATION_NOT_FOUND, "location": location}
            )

        result = data[0]

        # Extract location name from display_name or fallback to query
        location_name = result.get("display_name", location)

        # Nominatim returns lat/lon as strings, must convert to float
        lat = float(result.get("lat"))
        lon = float(result.get("lon"))

        geocoded = {
            "name": location_name,
            "latitude": lat,
            "longitude": lon,
            "country": result.get("address", {}).get("country"),
            "timezone": None  # Nominatim doesn't provide timezone
        }

        logger.info(f"Geocoded '{location}' to {location_name} ({lat}, {lon})")
        return geocoded

    except httpx.TimeoutException as e:
        logger.error(f"Geocoding timeout for '{location}': {e}")
        raise Exception(
            f"Geocoding request timed out for: {location}",
            {"error_code": ERROR_TIMEOUT, "location": location}
        )
    except ValueError:
        # Re-raise ValueError with original message
        raise
    except Exception as e:
        logger.error(f"Geocoding failed for '{location}': {e}")
        raise Exception(
            f"Geocoding service error: {str(e)}",
            {"error_code": ERROR_CODE_GEOCODE_SERVICE_ERROR, "location": location}
        )

# ============================================================================
# WEATHER FETCH HELPERS
# ============================================================================


def _safe_array_get(arr: List[Any], index: int, default: Any = None) -> Any:
    """Safely get item from array, returning default if index out of bounds.

    Args:
        arr: Array to access
        index: Index to retrieve
        default: Default value if index out of bounds

    Returns:
        Array item at index or default value
    """
    return arr[index] if index < len(arr) else default


def _build_base_params(lat: float, lon: float, units: str) -> Dict[str, Any]:
    """Build common API parameters for Open-Meteo requests.

    Args:
        lat: Latitude coordinate
        lon: Longitude coordinate
        units: Unit system ("metric" or "imperial")

    Returns:
        Dict with common API parameters
    """
    return {
        "latitude": lat,
        "longitude": lon,
        "temperature_unit": "fahrenheit" if units == "imperial" else "celsius",
        "wind_speed_unit": "mph" if units == "imperial" else "kmh",
        "timezone": "auto"
    }


async def _fetch_weather_data(
    params: Dict[str, Any],
    lat: float,
    lon: float,
    forecast_type: str
) -> Dict[str, Any]:
    """Fetch weather data from Open-Meteo API with error handling.

    Args:
        params: API request parameters
        lat: Latitude coordinate
        lon: Longitude coordinate
        forecast_type: Type of forecast being requested (for logging)

    Returns:
        Parsed JSON response from API

    Raises:
        Exception: If API request fails or times out
    """
    try:
        resp = await safe_http_get(WEATHER_API, params=params, timeout=API_TIMEOUT_SECONDS)
        return resp.json()
    except httpx.TimeoutException as e:
        logger.error(f"{forecast_type} weather fetch timeout for ({lat}, {lon}): {e}")
        raise Exception(
            f"Weather request timed out",
            {"error_code": ERROR_TIMEOUT, "latitude": lat, "longitude": lon}
        )
    except Exception as e:
        logger.error(f"Failed to fetch {forecast_type} weather for ({lat}, {lon}): {e}")
        raise Exception(
            f"Weather fetch failed: {str(e)}",
            {"error_code": ERROR_FETCH_FAILED, "latitude": lat, "longitude": lon}
        )


# ============================================================================
# WEATHER FETCH FUNCTIONS
# ============================================================================


async def fetch_current_weather(lat: float, lon: float, units: str = "metric") -> Dict[str, Any]:
    """Fetch current weather for given coordinates.

    Args:
        lat: Latitude coordinate
        lon: Longitude coordinate
        units: Unit system ("metric" or "imperial")

    Returns:
        Dict with current weather data including temperature, condition, humidity, wind speed,
        UV index, visibility, precipitation probability, cloud cover, and pressure

    Raises:
        Exception: If API request fails
    """
    logger.debug(f"Fetching current weather for ({lat}, {lon}) in {units}")

    params = _build_base_params(lat, lon, units)
    params["current"] = ",".join(CURRENT_WEATHER_PARAMS)

    data = await _fetch_weather_data(params, lat, lon, "current")
    current = data.get("current", {})

    weather_data = {
        "time": current.get("time"),
        "temperature": current.get("temperature_2m"),
        "feels_like": current.get("apparent_temperature"),
        "humidity": current.get("relative_humidity_2m"),
        "wind_speed": current.get("wind_speed_10m"),
        "condition": WMO_CODES.get(current.get("weather_code", 0), "Unknown"),
        "weather_code": current.get("weather_code"),
        "precipitation": current.get("precipitation"),
        "cloud_cover": current.get("cloud_cover"),
        "visibility": current.get("visibility"),
        "uv_index": current.get("uv_index"),
        "pressure_msl": current.get("pressure_msl"),
        "dew_point": current.get("dew_point_2m"),
        "units": units
    }

    logger.debug(f"Successfully fetched current weather: {weather_data['temperature']}{get_unit_symbol(units)}, {weather_data['condition']}")
    return weather_data


async def fetch_daily_forecast(lat: float, lon: float, units: str = "metric") -> Dict[str, Any]:
    """Fetch daily forecast (24 hours) for given coordinates.

    Args:
        lat: Latitude coordinate
        lon: Longitude coordinate
        units: Unit system ("metric" or "imperial")

    Returns:
        Dict with hourly forecast data for next 24 hours including temperature, humidity,
        precipitation, UV index, cloud cover, visibility, and wind

    Raises:
        Exception: If API request fails
    """
    logger.debug(f"Fetching daily forecast for ({lat}, {lon}) in {units}")

    params = _build_base_params(lat, lon, units)
    params["hourly"] = ",".join(HOURLY_FORECAST_PARAMS)

    data = await _fetch_weather_data(params, lat, lon, "daily")
    hourly = data.get("hourly", {})

    # Define field mapping for hourly forecast
    field_mapping = {
        "time": "time",
        "temperature": "temperature_2m",
        "humidity": "relative_humidity_2m",
        "weather_code": "weather_code",
        "wind_speed": "wind_speed_10m",
        "precipitation": "precipitation",
        "precipitation_probability": "precipitation_probability",
        "uv_index": "uv_index",
        "cloud_cover": "cloud_cover",
        "visibility": "visibility",
        "dew_point": "dew_point_2m",
    }

    # Build forecast items functionally (with conditions)
    forecast = _build_forecast_items(hourly, field_mapping, DAILY_FORECAST_HOURS, add_condition=True)

    logger.debug(f"Successfully fetched daily forecast with {len(forecast)} hourly entries")
    return {
        "forecast": forecast,
        "units": units
    }


async def fetch_weekly_forecast(lat: float, lon: float, units: str = "metric") -> Dict[str, Any]:
    """Fetch weekly forecast (7 days) for given coordinates.

    Args:
        lat: Latitude coordinate
        lon: Longitude coordinate
        units: Unit system ("metric" or "imperial")

    Returns:
        Dict with daily forecast data for next 7 days including temperature ranges,
        precipitation probability, UV index, sunrise/sunset, and wind

    Raises:
        Exception: If API request fails
    """
    logger.debug(f"Fetching weekly forecast for ({lat}, {lon}) in {units}")

    params = _build_base_params(lat, lon, units)
    params["daily"] = ",".join(DAILY_FORECAST_PARAMS)

    data = await _fetch_weather_data(params, lat, lon, "weekly")
    daily = data.get("daily", {})

    # Define field mapping for daily forecast
    field_mapping = {
        "date": "time",
        "temp_max": "temperature_2m_max",
        "temp_min": "temperature_2m_min",
        "weather_code": "weather_code",
        "wind_speed": "wind_speed_10m_max",
        "precipitation": "precipitation_sum",
        "precipitation_probability": "precipitation_probability_max",
        "uv_index": "uv_index_max",
        "sunrise": "sunrise",
        "sunset": "sunset",
    }

    # Build forecast items functionally (with conditions)
    forecast = _build_forecast_items(daily, field_mapping, WEEKLY_FORECAST_DAYS, add_condition=True)

    logger.debug(f"Successfully fetched weekly forecast with {len(forecast)} daily entries")
    return {
        "forecast": forecast,
        "units": units
    }

def _build_forecast_items(
    data_source: Dict[str, List[Any]],
    field_mapping: Dict[str, str],
    limit: int,
    add_condition: bool = True
) -> List[Dict[str, Any]]:
    """Build forecast items from API data using functional approach.

    Args:
        data_source: Dict containing arrays of weather data
        field_mapping: Maps output field names to API field names
        limit: Maximum number of items to return
        add_condition: If True, adds 'condition' field from weather_code

    Returns:
        List of forecast item dicts

    Example:
        field_mapping = {
            "time": "time",
            "temperature": "temperature_2m",
            "humidity": "relative_humidity_2m"
        }
    """
    # Extract and slice all arrays upfront
    arrays = {
        output_key: data_source.get(api_key, [])[:limit]
        for output_key, api_key in field_mapping.items()
    }

    # Determine the length from the first array (usually time)
    if not arrays:
        return []

    first_key = next(iter(arrays.keys()))
    num_items = len(arrays[first_key])

    # Build items using list comprehension
    items = [
        {key: _safe_array_get(arr, i) for key, arr in arrays.items()}
        for i in range(num_items)
    ]

    # Add condition from weather code if requested
    if add_condition:
        for item in items:
            item["condition"] = WMO_CODES.get(item.get("weather_code", 0), "Unknown")

    return items


# ============================================================================
# MCP TOOL DEFINITION
# ============================================================================


@mcp.tool()
async def get_weather(
    location: str,
    forecast_type: Literal["current", "daily", "weekly"] = "current",
    units: Literal["metric", "imperial"] = "imperial",
    ctx: Context = None
) -> CallToolResult:
    """Get weather information for a location.

    Uses Nominatim for geocoding and Open-Meteo for weather (no API keys required).
    Returns comprehensive weather data including temperature, conditions, precipitation,
    UV index, visibility, cloud cover, humidity, wind, and atmospheric pressure.

    Args:
        location: Location name (e.g., "Paris", "New York", "Tokyo")
        forecast_type: Type of forecast - "current", "daily" (24h), or "weekly" (7d)
        units: Unit system - "metric" (°C, km/h) or "imperial" (°F, mph)

    Returns:
        CallToolResult with formatted weather text, structured data, and rich metadata

    PARAMETER INTERPRETATION REFERENCE:
    Weather data includes numerical values that benefit from contextual interpretation:

    - UV Index: 0-2 (low risk), 3-5 (moderate), 6-7 (high), 8-10 (very high), 11+ (extreme)
      Protection needed at 3+, essential at 8+

    - Visibility: <1 km (poor/hazardous), 1-5 km (moderate), 5-10 km (good), >10 km (excellent)
      Below 5 km affects driving, below 1 km is dangerous

    - Humidity: 30-50% (comfortable), 50-70% (noticeable/sticky), >70% (uncomfortable), >85% (very humid/oppressive)
      High humidity makes temperatures feel more extreme

    - Wind Speed (km/h): <10 (calm), 10-30 (breezy/pleasant), 30-50 (windy/gusty), >50 (strong/difficult)
      Wind Speed (mph): <6 (calm), 6-19 (breezy/pleasant), 19-31 (windy/gusty), >31 (strong/difficult)

    - Precipitation Probability: <20% (unlikely), 20-50% (possible/uncertain), 50-80% (likely), >80% (very likely/expected)

    - Cloud Cover: 0-25% (clear/sunny), 25-50% (partly cloudy), 50-75% (mostly cloudy), 75-100% (overcast/grey)
      Affects UV exposure and temperature perception

    - Dew Point: Close to temperature indicates humid conditions, >20°C (68°F) feels oppressive,
      <10°C (50°F) feels comfortable and dry

    - Atmospheric Pressure: Rising pressure indicates improving weather, falling indicates worsening conditions
      Rapid changes often signal weather front movement

    USER CONTEXT:
    Users request weather information when planning outdoor activities, assessing travel conditions,
    determining appropriate clothing/preparation, or evaluating comfort and safety factors.
    They care about how conditions affect their specific plans, including:
    - Safety considerations (UV exposure, visibility for driving, severe weather warnings)
    - Comfort factors (how temperature actually feels with humidity/wind, oppressive conditions)
    - Activity planning (precipitation timing, suitable conditions for outdoor plans)
    - Preparation needs (sun protection, umbrellas, layers, travel delays)

    RESPONSE VOICE:
    Communicate weather like a knowledgeable local sharing practical information for planning:
    - Lead with the immediate experience ("It's a warm, muggy evening..." or "Cool and drizzly right now")
    - Highlight what matters for the user's likely intent (comfort, planning, safety)
    - Weave notable details naturally into conversation, not as a data checklist
    - Mention conditions that affect experience (high humidity making it feel hotter, excellent visibility
      for driving, low UV so no sunscreen needed, etc.)
    - Be conversational and contextual, not encyclopedic or formulaic

    Good examples of natural weather communication:
    - "It's a pleasant 72°F with light winds and clear skies. The low humidity makes it feel quite
      comfortable, and with excellent visibility you'd have great conditions for being outside."
    - "Currently 85°F but feels more like 91° with the high humidity - pretty sticky out there. Light
      winds aren't helping much. If you're heading out, definitely stay hydrated and bring sunscreen
      since the UV is high."
    - "Cool and drizzly at 56°F with reduced visibility around 5km. Roads might be slick, so take it
      easy if you're driving."
    - "Beautiful 68°F with low humidity and clear skies - ideal conditions. Gentle winds around 8mph.
      Great day for any outdoor plans."

    Examples:
        get_weather("Paris")
        get_weather("New York", forecast_type="daily", units="imperial")
        get_weather("Tokyo", forecast_type="weekly", units="metric")
    """
    request_timestamp = datetime.utcnow().isoformat() + "Z"
    logger.info(f"Weather request: location='{location}', type={forecast_type}, units={units}")

    if ctx:
        await ctx.info(f"Fetching {forecast_type} weather for {location}")

    try:
        # Validate location using shared validation
        is_valid, error_msg = validate_string_length(location, MIN_LOCATION_LENGTH, MAX_LOCATION_LENGTH, "location")
        if not is_valid:
            logger.warning(f"Location validation failed: {error_msg}")
            return create_validation_error(
                field_name="location",
                error_message=error_msg,
                field_value=location
            )

        # Validate forecast_type (already type-checked by Literal, but add runtime check)
        if forecast_type not in ["current", "daily", "weekly"]:
            logger.warning(f"Invalid forecast_type: {forecast_type}")
            return CallToolResult(
                content=[TextContent(type="text", text=f"Invalid forecast_type: {forecast_type}. Must be 'current', 'daily', or 'weekly'")],
                metadata={
                    "error_type": "validation_error",
                    "error_code": ERROR_CODE_INVALID_FORECAST_TYPE,
                    "forecast_type_provided": forecast_type,
                    "valid_options": ["current", "daily", "weekly"],
                    "timestamp": request_timestamp
                },
                isError=True
            )

        # Validate units (already type-checked by Literal, but add runtime check)
        if units not in ["metric", "imperial"]:
            logger.warning(f"Invalid units: {units}")
            return CallToolResult(
                content=[TextContent(type="text", text=f"Invalid units: {units}. Must be 'metric' or 'imperial'")],
                metadata={
                    "error_type": "validation_error",
                    "error_code": ERROR_CODE_INVALID_UNITS,
                    "units_provided": units,
                    "valid_options": ["metric", "imperial"],
                    "timestamp": request_timestamp
                },
                isError=True
            )

        # Geocode location
        if ctx:
            await ctx.report_progress(1, 5, f"Geocoding {location}...")

        try:
            geo_data = await geocode_location(location)
        except ValueError as e:
            # Location not found
            logger.warning(f"Location not found: {location}")
            if ctx:
                await ctx.warning(f"Location not found: {location}")
            return CallToolResult(
                content=[TextContent(type="text", text=f"Location not found: {location}")],
                metadata={
                    "error_type": "geocoding_error",
                    "error_code": ERROR_CODE_LOCATION_NOT_FOUND,
                    "location_query": location,
                    "timestamp": request_timestamp
                },
                isError=True
            )
        except Exception as e:
            # Geocoding service error
            logger.error(f"Geocoding service error for '{location}': {e}")
            if ctx:
                await ctx.error(f"Geocoding failed: {str(e)}")
            return CallToolResult(
                content=[TextContent(type="text", text=f"Geocoding service error: {str(e)}")],
                metadata={
                    "error_type": "geocoding_error",
                    "error_code": ERROR_CODE_GEOCODE_SERVICE_ERROR,
                    "location_query": location,
                    "timestamp": request_timestamp
                },
                isError=True
            )

        location_name = geo_data["name"]
        lat = geo_data["latitude"]
        lon = geo_data["longitude"]

        logger.info(f"Geocoded '{location}' to {location_name} ({lat}, {lon})")

        # Fetch appropriate weather data
        if ctx:
            await ctx.report_progress(2, 5, f"Fetching {forecast_type} weather...")

        try:
            if forecast_type == "current":
                weather_data = await fetch_current_weather(lat, lon, units)
            elif forecast_type == "daily":
                weather_data = await fetch_daily_forecast(lat, lon, units)
            else:  # weekly
                weather_data = await fetch_weekly_forecast(lat, lon, units)
        except Exception as e:
            # Weather fetch error
            logger.error(f"Weather fetch failed for {location_name}: {e}")
            if ctx:
                await ctx.error(f"Weather fetch failed: {str(e)}")

            # Determine error code from exception metadata if available
            error_code = ERROR_CODE_FETCH_FAILED
            if hasattr(e, 'args') and len(e.args) > 1 and isinstance(e.args[1], dict):
                error_code = e.args[1].get("error_code", ERROR_CODE_FETCH_FAILED)

            return CallToolResult(
                content=[TextContent(type="text", text=f"Weather fetch failed: {str(e)}")],
                metadata={
                    "error_type": "fetch_error",
                    "error_code": error_code,
                    "location": location_name,
                    "latitude": lat,
                    "longitude": lon,
                    "forecast_type": forecast_type,
                    "timestamp": request_timestamp
                },
                isError=True
            )

        # Prepare response
        if ctx:
            await ctx.report_progress(4, 5, "Preparing weather data...")

        # Build rich metadata for response
        metadata = {
            "location": location_name,
            "latitude": lat,
            "longitude": lon,
            "country": geo_data.get("country"),
            "timezone": geo_data.get("timezone"),
            "forecast_type": forecast_type,
            "units": units,
            "timestamp": request_timestamp,
            "data": weather_data
        }

        if ctx:
            await ctx.report_progress(5, 5, "Complete")

        logger.info(f"Successfully retrieved {forecast_type} weather for {location_name}")

        # Return raw JSON data, let model interpret using the voice guidance
        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps(weather_data, indent=2))],
            metadata=metadata
        )

    except Exception as e:
        # Catch-all for unexpected errors
        logger.error(f"Unexpected error in get_weather: {e}", exc_info=True)
        if ctx:
            await ctx.error(f"Unexpected error: {str(e)}")

        return CallToolResult(
            content=[TextContent(type="text", text=f"Unexpected error: {str(e)}")],
            metadata={
                "error_type": "unexpected_error",
                "error_code": "unknown_error",
                "location_query": location,
                "forecast_type": forecast_type,
                "units": units,
                "timestamp": request_timestamp
            },
            isError=True
        )

# ============================================================================
# SERVER ENTRY POINT
# ============================================================================


if __name__ == "__main__":
    logger.info("Starting Weather MCP server (Streamable HTTP)...")
    server.run(transport="streamable-http")
