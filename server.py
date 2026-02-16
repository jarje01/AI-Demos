"""
Simple MCP Server Demo

This is a basic MCP server that demonstrates the Model Context Protocol.
It provides a few simple tools for demonstration purposes.
"""

import asyncio
from typing import Any
from mcp.server.fastmcp import FastMCP

# Create the MCP server
server = FastMCP("demo-server")

@server.tool()
async def add_numbers(a: int, b: int) -> str:
    """Add two numbers together.

    Args:
        a: First number
        b: Second number

    Returns:
        The sum of the two numbers
    """
    result = a + b
    return f"{a} + {b} = {result}"

@server.tool()
async def multiply_numbers(x: float, y: float) -> str:
    """Multiply two numbers together.

    Args:
        x: First number
        y: Second number

    Returns:
        The product of the two numbers
    """
    result = x * y
    return f"{x} × {y} = {result}"

@server.tool()
async def get_current_time() -> str:
    """Get the current date and time.

    Returns:
        Current date and time in ISO format
    """
    from datetime import datetime
    now = datetime.now()
    return f"Current time: {now.isoformat()}"

if __name__ == "__main__":
    server.run(transport="stdio")