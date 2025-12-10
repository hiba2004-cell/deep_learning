import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { CallToolRequestSchema, ListToolsRequestSchema, } from "@modelcontextprotocol/sdk/types.js";
import { query } from "./db.js";
// import { query } from "./db";
const server = new Server({ name: "mysql-mcp-server", version: "1.0.0" }, { capabilities: { tools: {} } });
// List available tools
server.setRequestHandler(ListToolsRequestSchema, async () => ({
    tools: [
        {
            name: "query_database",
            description: "Execute a SQL query on the MySQL database",
            inputSchema: {
                type: "object",
                properties: {
                    sql: { type: "string", description: "The SQL query to execute" },
                    params: {
                        type: "array",
                        description: "Optional parameters for the query",
                        items: { type: "string" }
                    }
                },
                required: ["sql"]
            }
        }
    ]
}));
// Handle tool calls
server.setRequestHandler(CallToolRequestSchema, async (request) => {
    if (request.params.name === "query_database") {
        const { sql, params } = request.params.arguments;
        try {
            const results = await query(sql, params);
            return { content: [{ type: "text", text: JSON.stringify(results, null, 2) }] };
        }
        catch (error) {
            return { content: [{ type: "text", text: `Error executing query: ${error.message}` }], isError: true };
        }
    }
    throw new Error(`Unknown tool: ${request.params.name}`);
});
// Start server
async function main() {
    const transport = new StdioServerTransport();
    await server.connect(transport);
    console.error("MySQL MCP Server running on stdio");
}
main().catch(console.error);
