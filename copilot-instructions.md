# MCP Server Demo Project Setup

## Project Overview
This is a demonstration MCP (Model Context Protocol) server built with Python. It showcases basic MCP server implementation with tools for arithmetic operations and time retrieval.

## Setup Checklist

### ✅ Project Structure Created
- [x] MCP directory created
- [x] server.py - Main MCP server implementation
- [x] pyproject.toml - Python project configuration
- [x] README.md - Project documentation
- [x] .vscode/mcp.json - VS Code MCP configuration

### 🔧 Dependencies
- [ ] Install MCP Python SDK: `pip install mcp`
- [ ] Install project in development mode: `pip install -e .`

### 🧪 Testing
- [ ] Test server startup: `python server.py`
- [ ] Verify tools are available (add_numbers, multiply_numbers, get_current_time)
- [ ] Test with MCP Inspector: `npx @modelcontextprotocol/inspector python server.py`

### 🔌 Integration
- [ ] Configure with Claude Desktop or other MCP client
- [ ] Test tool execution from client
- [ ] Verify server responses

## MCP Server Features

### Tools Provided
1. **add_numbers(a: int, b: int)** - Adds two integers
2. **multiply_numbers(x: float, y: float)** - Multiplies two floats
3. **get_current_time()** - Returns current timestamp

### Configuration
The server uses stdio transport for communication with MCP clients. It's configured to run as a simple Python script.

## Development Notes

- Uses the official MCP Python SDK
- Implements async/await patterns for tool execution
- Follows MCP protocol specifications for tool definitions and responses
- Includes proper error handling and type hints

## Next Steps

1. Install dependencies and test the server
2. Integrate with an MCP client (Claude Desktop, VS Code, etc.)
3. Extend with additional tools or resources as needed
4. Explore advanced MCP features like resources and prompts