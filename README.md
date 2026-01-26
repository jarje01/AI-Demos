# MCP Demo Server

A simple demonstration of the Model Context Protocol (MCP) server implementation in Python.

## Features

This MCP server provides three basic tools:

- **add_numbers**: Adds two integers together
- **multiply_numbers**: Multiplies two floating-point numbers
- **get_current_time**: Returns the current date and time

## Installation

1. Install the required dependencies:
```bash
pip install -e .
```

Or using uv:
```bash
uv pip install -e .
```

## Usage

Run the server:
```bash
python server.py
```

Or if installed as a package:
```bash
mcp-demo-server
```

## MCP Configuration

To use this server with an MCP client (like Claude Desktop), add it to your configuration:

```json
{
  "mcpServers": {
    "demo": {
      "command": "python",
      "args": ["C:\\path\\to\\MCP\\server.py"]
    }
  }
}
```

Replace `C:\\path\\to\\MCP\\server.py` with the actual path to the server.py file.

## Development

This server uses the official MCP Python SDK. For more information about MCP, visit:
https://modelcontextprotocol.io/