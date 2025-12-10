import asyncio
import json
import sys
from typing import Any
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
from db import query

# Create server instance
server = Server("mysql-mcp-server")

@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools"""
    return [
        Tool(
            name="query_database",
            description="Execute a SQL query on the MySQL database",
            inputSchema={
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": "The SQL query to execute"
                    },
                    "params": {
                        "type": "array",
                        "description": "Optional parameters for the query",
                        "items": {"type": "string"}
                    }
                },
                "required": ["sql"]
            }
        )
    ]

@server.call_tool()
async def call_tool(name: str, arguments: Any) -> list[TextContent]:
    """Handle tool calls"""
    if name == "query_database":
        sql = arguments.get("sql")
        params = arguments.get("params")
        
        try:
            results = await query(sql, params)
            return [
                TextContent(
                    type="text",
                    text=json.dumps(results, indent=2, default=str)
                )
            ]
        except Exception as error:
            return [
                TextContent(
                    type="text",
                    text=f"Error executing query: {str(error)}"
                )
            ]
    
    raise ValueError(f"Unknown tool: {name}")

async def main():
    """Main entry point to run the server"""
    async with stdio_server() as (read_stream, write_stream):
        print("MySQL MCP Server running on stdio", file=sys.stderr)
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )

if __name__ == "__main__":
    asyncio.run(main())