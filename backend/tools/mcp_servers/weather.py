"""ABOUTME: Weather MCP Server - Open-Meteo based weather tool with current, daily, and weekly forecasts.

Provides unified weather data via Open-Meteo (no API key required). Returns dual-format
responses: human-readable text for LLM context + structured JSON for frontend rendering.
"""

import asyncio
import json
import logging
import os
from typing import Optional, Dict, Any
from datetime import datetime, timedelta

import httpx
from mcp.server.fastmcp import Context

from common.mcp_base import MCPServerBase

# Initialize MCP server with base class
server = MCPServerBase("weather")
mcp = server.get_mcp()
logger = server.get_logger()

# Open-Meteo API endpoints
GEOCODING_API = "https://geocoding-api.open-meteo.com/v1/search"
WEATHER_API = "https://api.open-meteo.com/v1/forecast"

# WMO Weather codes mapping
WMO_CODES = {
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


def get_weather_emoji(code: int) -> str:
    """Get emoji for WMO weather code.

    Args:
        code: WMO weather code

    Returns:
        Emoji representing the weather condition
    """
    if code == 0:
        return "☀️"
    elif code in [1, 2]:
        return "⛅"
    elif code == 3:
        return "☁️"
    elif code in [45, 48]:
        return "🌫️"
    elif code in [51, 53, 55, 80, 81, 82]:
        return "🌧️"
    elif code in [61, 63, 65]:
        return "🌧️"
    elif code in [71, 73, 75, 77, 85, 86]:
        return "❄️"
    elif code in [95, 96, 99]:
        return "⛈️"
    else:
        return "🌤️"


async def geocode_location(location: str) -> Optional[Dict[str, Any]]:
    """Convert location name to coordinates using Open-Meteo Geocoding API.

    Args:
        location: Location name (e.g., "Paris", "New York", "Tokyo")

    Returns:
        Dict with 'name', 'latitude', 'longitude', 'country' or None if not found

    Raises:
        Exception: If API request fails
    """
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            params = {
                "name": location,
                "count": 1,
                "language": "en",
                "format": "json"
            }
            resp = await client.get(GEOCODING_API, params=params)
            if resp.status_code != 200:
                raise Exception(f"Geocoding API returned {resp.status_code}")

            data = resp.json()
            results = data.get("results", [])

            if not results:
                return None

            result = results[0]
            return {
                "name": f"{result.get('name', location)}, {result.get('country', '')}".strip(),
                "latitude": result.get("latitude"),
                "longitude": result.get("longitude"),
                "country": result.get("country"),
                "timezone": result.get("timezone")
            }
    except Exception as e:
        logger.error(f"Geocoding failed for '{location}': {e}")
        raise


async def fetch_current_weather(lat: float, lon: float, units: str = "celsius") -> Dict[str, Any]:
    """Fetch current weather for given coordinates.

    Args:
        lat: Latitude
        lon: Longitude
        units: "celsius" or "fahrenheit"

    Returns:
        Dict with current weather data
    """
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            params = {
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m,apparent_temperature",
                "temperature_unit": "fahrenheit" if units == "fahrenheit" else "celsius",
                "wind_speed_unit": "kmh",
                "timezone": "auto"
            }
            resp = await client.get(WEATHER_API, params=params)
            if resp.status_code != 200:
                error_text = resp.text
                logger.error(f"Weather API returned {resp.status_code}: {error_text}")
                raise Exception(f"Weather API returned {resp.status_code}: {error_text}")

            data = resp.json()
            current = data.get("current", {})

            return {
                "time": current.get("time"),
                "temperature": current.get("temperature_2m"),
                "feels_like": current.get("apparent_temperature"),
                "humidity": current.get("relative_humidity_2m"),
                "wind_speed": current.get("wind_speed_10m"),
                "condition": WMO_CODES.get(current.get("weather_code", 0), "Unknown"),
                "weather_code": current.get("weather_code"),
                "units": units
            }
    except Exception as e:
        logger.error(f"Failed to fetch current weather: {e}")
        raise


async def fetch_daily_forecast(lat: float, lon: float, units: str = "celsius") -> Dict[str, Any]:
    """Fetch daily forecast (24 hours) for given coordinates.

    Args:
        lat: Latitude
        lon: Longitude
        units: "celsius" or "fahrenheit"

    Returns:
        Dict with daily forecast data
    """
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            params = {
                "latitude": lat,
                "longitude": lon,
                "hourly": "temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m,precipitation",
                "temperature_unit": "fahrenheit" if units == "fahrenheit" else "celsius",
                "wind_speed_unit": "kmh",
                "timezone": "auto"
            }
            resp = await client.get(WEATHER_API, params=params)
            if resp.status_code != 200:
                error_text = resp.text
                logger.error(f"Weather API returned {resp.status_code}: {error_text}")
                raise Exception(f"Weather API returned {resp.status_code}: {error_text}")

            data = resp.json()
            hourly = data.get("hourly", {})

            # Get next 24 hours
            times = hourly.get("time", [])[:24]
            temps = hourly.get("temperature_2m", [])[:24]
            humidity = hourly.get("relative_humidity_2m", [])[:24]
            codes = hourly.get("weather_code", [])[:24]
            wind = hourly.get("wind_speed_10m", [])[:24]
            precip = hourly.get("precipitation", [])[:24]

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

            return {
                "forecast": forecast,
                "units": units
            }
    except Exception as e:
        logger.error(f"Failed to fetch daily forecast: {e}")
        raise


async def fetch_weekly_forecast(lat: float, lon: float, units: str = "celsius") -> Dict[str, Any]:
    """Fetch weekly forecast (7 days) for given coordinates.

    Args:
        lat: Latitude
        lon: Longitude
        units: "celsius" or "fahrenheit"

    Returns:
        Dict with weekly forecast data
    """
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            params = {
                "latitude": lat,
                "longitude": lon,
                "daily": "temperature_2m_max,temperature_2m_min,weather_code,wind_speed_10m_max,precipitation_sum",
                "temperature_unit": "fahrenheit" if units == "fahrenheit" else "celsius",
                "wind_speed_unit": "kmh",
                "timezone": "auto"
            }
            resp = await client.get(WEATHER_API, params=params)
            if resp.status_code != 200:
                error_text = resp.text
                logger.error(f"Weather API returned {resp.status_code}: {error_text}")
                raise Exception(f"Weather API returned {resp.status_code}: {error_text}")

            data = resp.json()
            daily = data.get("daily", {})

            # Get next 7 days
            times = daily.get("time", [])[:7]
            temps_max = daily.get("temperature_2m_max", [])[:7]
            temps_min = daily.get("temperature_2m_min", [])[:7]
            codes = daily.get("weather_code", [])[:7]
            wind = daily.get("wind_speed_10m_max", [])[:7]
            precip = daily.get("precipitation_sum", [])[:7]

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

            return {
                "forecast": forecast,
                "units": units
            }
    except Exception as e:
        logger.error(f"Failed to fetch weekly forecast: {e}")
        raise


def format_current_weather_text(location: str, data: Dict[str, Any]) -> str:
    """Format current weather data as human-readable text.

    Args:
        location: Location name
        data: Current weather data dict

    Returns:
        Formatted text for LLM
    """
    emoji = get_weather_emoji(data.get("weather_code", 0))
    temp = data.get("temperature")
    condition = data.get("condition")
    feels_like = data.get("feels_like")
    humidity = data.get("humidity")
    wind = data.get("wind_speed")

    text = f"**Current weather in {location}:**\n\n"
    text += f"{emoji} **{condition}**\n"
    text += f"Temperature: {temp}°C (feels like {feels_like}°C)\n"
    text += f"Humidity: {humidity}%\n"
    text += f"Wind: {wind} km/h\n"

    return text


def format_daily_forecast_text(location: str, data: Dict[str, Any]) -> str:
    """Format daily forecast as human-readable text.

    Args:
        location: Location name
        data: Daily forecast data dict

    Returns:
        Formatted text for LLM
    """
    forecast = data.get("forecast", [])
    text = f"**24-hour forecast for {location}:**\n\n"

    # Show hourly data for today
    for item in forecast[:8]:  # Show every 3rd hour (8 entries = 24 hours)
        time = item.get("time", "").split("T")[1] if item.get("time") else "?"
        temp = item.get("temperature")
        condition = item.get("condition")
        emoji = get_weather_emoji(item.get("weather_code", 0))

        text += f"{time}: {emoji} {temp}°C - {condition}\n"

    return text


def format_weekly_forecast_text(location: str, data: Dict[str, Any]) -> str:
    """Format weekly forecast as human-readable text.

    Args:
        location: Location name
        data: Weekly forecast data dict

    Returns:
        Formatted text for LLM
    """
    forecast = data.get("forecast", [])
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

        text += f"{day_name}: {emoji} {temp_max}°C / {temp_min}°C - {condition}\n"

    return text


@mcp.tool()
async def get_weather(
    location: str,
    forecast_type: str = "current",
    units: str = "fahrenheit",
    ctx: Context = None
) -> dict:
    """Get weather information for a location.

    Supports current weather and forecasts using Open-Meteo (no API key required).

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
                      (default: "current")
        units: Temperature units - "celsius" or "fahrenheit" (default: "fahrenheit")
        ctx: MCP context for progress/logging (auto-injected)

    Returns:
        Dict with weather data and structured format for UI rendering

    Examples:
        # Current weather
        get_weather("Paris")

        # Daily forecast in Fahrenheit
        get_weather("New York", forecast_type="daily", units="fahrenheit")

        # Weekly forecast
        get_weather("Tokyo", forecast_type="weekly")
    """
    logger.info(f"Getting {forecast_type} weather for {location} ({units})")
    if ctx:
        await ctx.info(f"Fetching {forecast_type} weather for {location}")

    try:
        # Validate forecast_type
        if forecast_type not in ["current", "daily", "weekly"]:
            raise ValueError(f"Invalid forecast_type: {forecast_type}. Must be 'current', 'daily', or 'weekly'")

        # Validate units
        if units not in ["celsius", "fahrenheit"]:
            raise ValueError(f"Invalid units: {units}. Must be 'celsius' or 'fahrenheit'")

        # Geocode location
        if ctx:
            await ctx.report_progress(1, 5, f"Geocoding {location}...")

        geo_data = await geocode_location(location)
        if not geo_data:
            raise ValueError(f"Location not found: {location}")

        location_name = geo_data["name"]
        lat = geo_data["latitude"]
        lon = geo_data["longitude"]

        logger.info(f"Geocoded {location} to {location_name} ({lat}, {lon})")

        # Fetch appropriate weather data
        if forecast_type == "current":
            if ctx:
                await ctx.report_progress(2, 5, "Fetching current weather...")
            weather_data = await fetch_current_weather(lat, lon, units)
        elif forecast_type == "daily":
            if ctx:
                await ctx.report_progress(2, 5, "Fetching daily forecast...")
            weather_data = await fetch_daily_forecast(lat, lon, units)
        else:  # weekly
            if ctx:
                await ctx.report_progress(2, 5, "Fetching weekly forecast...")
            weather_data = await fetch_weekly_forecast(lat, lon, units)

        # Build response with simplified data for LLM + structured data for UI
        response = {
            "location": location_name,
            "type": forecast_type,
            "units": units,
            "data": weather_data,
            # Structured format for WeatherCard UI component
            "weather_data": {
                "location": location_name,
                "latitude": lat,
                "longitude": lon,
                "type": forecast_type,
                "units": units,
                "data": weather_data
            }
        }

        if ctx:
            await ctx.report_progress(5, 5, "Complete")

        logger.info(f"Successfully retrieved {forecast_type} weather for {location_name}")
        return response

    except ValueError as e:
        logger.warning(f"Invalid input: {e}")
        if ctx:
            await ctx.warning(f"Invalid input: {e}")
        error_response = {
            "content": f"Error: {str(e)}",
            "weather_data": None
        }
        return error_response
    except Exception as e:
        logger.error(f"Failed to get weather: {e}", exc_info=True)
        if ctx:
            await ctx.error(f"Failed to get weather: {str(e)}")
        error_response = {
            "content": f"Error fetching weather: {str(e)}",
            "weather_data": None
        }
        return error_response


if __name__ == "__main__":
    logger.info("Starting Weather MCP server (Streamable HTTP)...")
    mcp.run(transport="streamable-http")
