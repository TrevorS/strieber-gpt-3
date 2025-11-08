"""ABOUTME: Weather MCP Server - Open-Meteo weather data with Nominatim geocoding.

Provides weather data via Open-Meteo API (no API key required). Uses OSM Nominatim for
geocoding to handle location names like "Savannah, GA". Returns dual-format responses:
human-readable text for LLM context + structured JSON for frontend rendering.
"""

import asyncio
import json
import logging
import os
from typing import Optional, Dict, Any, Literal
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

class TemperatureUnits(str, Enum):
    """Valid temperature units."""
    CELSIUS = "celsius"
    FAHRENHEIT = "fahrenheit"

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
    units: Literal["celsius", "fahrenheit"] = Field(
        default="fahrenheit",
        description="Temperature units - 'celsius' or 'fahrenheit'"
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
        units: Temperature units ("celsius" or "fahrenheit")

    Returns:
        Unit symbol string
    """
    return FAHRENHEIT_DISPLAY_UNIT if units == "fahrenheit" else CELSIUS_DISPLAY_UNIT

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
# WEATHER FETCH FUNCTIONS
# ============================================================================


async def fetch_current_weather(lat: float, lon: float, units: str = "celsius") -> Dict[str, Any]:
    """Fetch current weather for given coordinates.

    Args:
        lat: Latitude coordinate
        lon: Longitude coordinate
        units: Temperature units ("celsius" or "fahrenheit")

    Returns:
        Dict with current weather data including temperature, condition, humidity, wind speed

    Raises:
        Exception: If API request fails
    """
    logger.debug(f"Fetching current weather for ({lat}, {lon}) in {units}")

    try:
        params = {
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m,apparent_temperature",
            "temperature_unit": "fahrenheit" if units == "fahrenheit" else "celsius",
            "wind_speed_unit": "kmh",
            "timezone": "auto"
        }
        resp = await safe_http_get(WEATHER_API, params=params, timeout=API_TIMEOUT_SECONDS)

        data = resp.json()
        current = data.get("current", {})

        weather_data = {
            "time": current.get("time"),
            "temperature": current.get("temperature_2m"),
            "feels_like": current.get("apparent_temperature"),
            "humidity": current.get("relative_humidity_2m"),
            "wind_speed": current.get("wind_speed_10m"),
            "condition": WMO_CODES.get(current.get("weather_code", 0), "Unknown"),
            "weather_code": current.get("weather_code"),
            "units": units
        }

        logger.debug(f"Successfully fetched current weather: {weather_data['temperature']}{get_unit_symbol(units)}, {weather_data['condition']}")
        return weather_data

    except httpx.TimeoutException as e:
        logger.error(f"Weather fetch timeout for ({lat}, {lon}): {e}")
        raise Exception(
            f"Weather request timed out",
            {"error_code": ERROR_TIMEOUT, "latitude": lat, "longitude": lon}
        )
    except Exception as e:
        logger.error(f"Failed to fetch current weather: {e}")
        raise Exception(
            f"Weather fetch failed: {str(e)}",
            {"error_code": ERROR_FETCH_FAILED, "latitude": lat, "longitude": lon}
        )


async def fetch_daily_forecast(lat: float, lon: float, units: str = "celsius") -> Dict[str, Any]:
    """Fetch daily forecast (24 hours) for given coordinates.

    Args:
        lat: Latitude coordinate
        lon: Longitude coordinate
        units: Temperature units ("celsius" or "fahrenheit")

    Returns:
        Dict with hourly forecast data for next 24 hours

    Raises:
        Exception: If API request fails
    """
    logger.debug(f"Fetching daily forecast for ({lat}, {lon}) in {units}")

    try:
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": "temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m,precipitation",
            "temperature_unit": "fahrenheit" if units == "fahrenheit" else "celsius",
            "wind_speed_unit": "kmh",
            "timezone": "auto"
        }
        resp = await safe_http_get(WEATHER_API, params=params, timeout=API_TIMEOUT_SECONDS)

        data = resp.json()
        hourly = data.get("hourly", {})

        # Get next 24 hours
        times = hourly.get("time", [])[:DAILY_FORECAST_HOURS]
        temps = hourly.get("temperature_2m", [])[:DAILY_FORECAST_HOURS]
        humidity = hourly.get("relative_humidity_2m", [])[:DAILY_FORECAST_HOURS]
        codes = hourly.get("weather_code", [])[:DAILY_FORECAST_HOURS]
        wind = hourly.get("wind_speed_10m", [])[:DAILY_FORECAST_HOURS]
        precip = hourly.get("precipitation", [])[:DAILY_FORECAST_HOURS]

        forecast = []
        for i, time_str in enumerate(times):
            forecast.append({
                "time": time_str,
                "temperature": temps[i] if i < len(temps) else None,
                "humidity": humidity[i] if i < len(humidity) else None,
                "condition": WMO_CODES.get(codes[i] if i < len(codes) else 0, "Unknown"),
                "weather_code": codes[i] if i < len(codes) else None,
                "wind_speed": wind[i] if i < len(wind) else None,
                "precipitation": precip[i] if i < len(precip) else None,
            })

        logger.debug(f"Successfully fetched daily forecast with {len(forecast)} hourly entries")
        return {
            "forecast": forecast,
            "units": units
        }

    except httpx.TimeoutException as e:
        logger.error(f"Daily forecast timeout for ({lat}, {lon}): {e}")
        raise Exception(
            f"Weather request timed out",
            {"error_code": ERROR_TIMEOUT, "latitude": lat, "longitude": lon}
        )
    except Exception as e:
        logger.error(f"Failed to fetch daily forecast: {e}")
        raise Exception(
            f"Weather fetch failed: {str(e)}",
            {"error_code": ERROR_FETCH_FAILED, "latitude": lat, "longitude": lon}
        )


async def fetch_weekly_forecast(lat: float, lon: float, units: str = "celsius") -> Dict[str, Any]:
    """Fetch weekly forecast (7 days) for given coordinates.

    Args:
        lat: Latitude coordinate
        lon: Longitude coordinate
        units: Temperature units ("celsius" or "fahrenheit")

    Returns:
        Dict with daily forecast data for next 7 days

    Raises:
        Exception: If API request fails
    """
    logger.debug(f"Fetching weekly forecast for ({lat}, {lon}) in {units}")

    try:
        params = {
            "latitude": lat,
            "longitude": lon,
            "daily": "temperature_2m_max,temperature_2m_min,weather_code,wind_speed_10m_max,precipitation_sum",
            "temperature_unit": "fahrenheit" if units == "fahrenheit" else "celsius",
            "wind_speed_unit": "kmh",
            "timezone": "auto"
        }
        resp = await safe_http_get(WEATHER_API, params=params, timeout=API_TIMEOUT_SECONDS)

        data = resp.json()
        daily = data.get("daily", {})

        # Get next 7 days
        times = daily.get("time", [])[:WEEKLY_FORECAST_DAYS]
        temps_max = daily.get("temperature_2m_max", [])[:WEEKLY_FORECAST_DAYS]
        temps_min = daily.get("temperature_2m_min", [])[:WEEKLY_FORECAST_DAYS]
        codes = daily.get("weather_code", [])[:WEEKLY_FORECAST_DAYS]
        wind = daily.get("wind_speed_10m_max", [])[:WEEKLY_FORECAST_DAYS]
        precip = daily.get("precipitation_sum", [])[:WEEKLY_FORECAST_DAYS]

        forecast = []
        for i, time_str in enumerate(times):
            forecast.append({
                "date": time_str,
                "temp_max": temps_max[i] if i < len(temps_max) else None,
                "temp_min": temps_min[i] if i < len(temps_min) else None,
                "condition": WMO_CODES.get(codes[i] if i < len(codes) else 0, "Unknown"),
                "weather_code": codes[i] if i < len(codes) else None,
                "wind_speed": wind[i] if i < len(wind) else None,
                "precipitation": precip[i] if i < len(precip) else None,
            })

        logger.debug(f"Successfully fetched weekly forecast with {len(forecast)} daily entries")
        return {
            "forecast": forecast,
            "units": units
        }

    except httpx.TimeoutException as e:
        logger.error(f"Weekly forecast timeout for ({lat}, {lon}): {e}")
        raise Exception(
            f"Weather request timed out",
            {"error_code": ERROR_TIMEOUT, "latitude": lat, "longitude": lon}
        )
    except Exception as e:
        logger.error(f"Failed to fetch weekly forecast: {e}")
        raise Exception(
            f"Weather fetch failed: {str(e)}",
            {"error_code": ERROR_FETCH_FAILED, "latitude": lat, "longitude": lon}
        )

# ============================================================================
# TEXT FORMATTING FUNCTIONS
# ============================================================================


def format_current_weather_text(location: str, data: Dict[str, Any]) -> str:
    """Format current weather data as human-readable text.

    Args:
        location: Location name
        data: Current weather data dict

    Returns:
        Formatted markdown text for LLM
    """
    emoji = get_weather_emoji(data.get("weather_code", 0))
    temp = data.get("temperature")
    condition = data.get("condition")
    feels_like = data.get("feels_like")
    humidity = data.get("humidity")
    wind = data.get("wind_speed")
    units = data.get("units", "celsius")
    unit_symbol = get_unit_symbol(units)

    text = f"**Current weather in {location}:**\n\n"
    text += f"{emoji} **{condition}**\n"
    text += f"Temperature: {temp}{unit_symbol} (feels like {feels_like}{unit_symbol})\n"
    text += f"Humidity: {humidity}%\n"
    text += f"Wind: {wind} km/h\n"

    return text


def format_daily_forecast_text(location: str, data: Dict[str, Any]) -> str:
    """Format daily forecast as human-readable text.

    Args:
        location: Location name
        data: Daily forecast data dict

    Returns:
        Formatted markdown text for LLM
    """
    forecast = data.get("forecast", [])
    units = data.get("units", "celsius")
    unit_symbol = get_unit_symbol(units)
    text = f"**24-hour forecast for {location}:**\n\n"

    # Show every 3rd hour (8 entries = 24 hours)
    step = HOURLY_DISPLAY_INTERVAL
    for i in range(0, len(forecast), step):
        if i >= len(forecast):
            break
        item = forecast[i]
        time = item.get("time", "").split("T")[1] if item.get("time") else "?"
        temp = item.get("temperature")
        condition = item.get("condition")
        emoji = get_weather_emoji(item.get("weather_code", 0))

        text += f"{time}: {emoji} {temp}{unit_symbol} - {condition}\n"

    return text


def format_weekly_forecast_text(location: str, data: Dict[str, Any]) -> str:
    """Format weekly forecast as human-readable text.

    Args:
        location: Location name
        data: Weekly forecast data dict

    Returns:
        Formatted markdown text for LLM
    """
    forecast = data.get("forecast", [])
    units = data.get("units", "celsius")
    unit_symbol = get_unit_symbol(units)
    text = f"**7-day forecast for {location}:**\n\n"

    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    for i, item in enumerate(forecast):
        date_str = item.get("date", "")
        if date_str:
            date_obj = datetime.fromisoformat(date_str)
            day_name = day_names[date_obj.weekday()]
        else:
            day_name = "?"

        temp_max = item.get("temp_max")
        temp_min = item.get("temp_min")
        condition = item.get("condition")
        emoji = get_weather_emoji(item.get("weather_code", 0))

        text += f"{day_name}: {emoji} {temp_max}{unit_symbol} / {temp_min}{unit_symbol} - {condition}\n"

    return text

# ============================================================================
# MCP TOOL DEFINITION
# ============================================================================


@mcp.tool()
async def get_weather(
    location: str,
    forecast_type: Literal["current", "daily", "weekly"] = "current",
    units: Literal["celsius", "fahrenheit"] = "fahrenheit",
    ctx: Context = None
) -> CallToolResult:
    """Get weather information for a location.

    Uses Nominatim for geocoding and Open-Meteo for weather (no API keys required).

    IMPORTANT RESPONSE GUIDANCE:
    After receiving the result, provide a brief, conversational weather summary:
    - For CURRENT: Mention temperature, condition, and one detail (humidity/wind/feels-like)
    - For DAILY: Mention high/low temps and expected conditions through the day
    - For WEEKLY: Highlight the best and worst days in the forecast
    - Keep responses to 1-2 sentences maximum
    - Do NOT repeat or format the raw data output - use it to inform your response
    - Example: "It's 56°F in Tokyo with drizzle, quite humid at 96%"

    Args:
        location: Location name (e.g., "Paris", "New York", "Tokyo")
        forecast_type: Type of forecast - "current", "daily" (24h), or "weekly" (7d)
        units: Temperature units - "celsius" or "fahrenheit"

    Returns:
        CallToolResult with formatted weather text, structured data, and rich metadata

    Examples:
        # Current weather
        get_weather("Paris")

        # Daily forecast in Fahrenheit
        get_weather("New York", forecast_type="daily", units="fahrenheit")

        # Weekly forecast
        get_weather("Tokyo", forecast_type="weekly")
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
        if units not in ["celsius", "fahrenheit"]:
            logger.warning(f"Invalid units: {units}")
            return CallToolResult(
                content=[TextContent(type="text", text=f"Invalid units: {units}. Must be 'celsius' or 'fahrenheit'")],
                metadata={
                    "error_type": "validation_error",
                    "error_code": ERROR_CODE_INVALID_UNITS,
                    "units_provided": units,
                    "valid_options": ["celsius", "fahrenheit"],
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

        # Format weather text for LLM
        if ctx:
            await ctx.report_progress(4, 5, "Formatting weather data...")

        if forecast_type == "current":
            formatted_text = format_current_weather_text(location_name, weather_data)
        elif forecast_type == "daily":
            formatted_text = format_daily_forecast_text(location_name, weather_data)
        else:  # weekly
            formatted_text = format_weekly_forecast_text(location_name, weather_data)

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

        return CallToolResult(
            content=[TextContent(type="text", text=formatted_text)],
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
