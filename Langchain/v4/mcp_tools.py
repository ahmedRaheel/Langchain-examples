from langchain_mcp_adapters.client import MultiServerMCPClient
client = MultiServerMCPClient(
    {
        "time_server": {
            "transport": "stdio",
            "command": "python",
            "args": [
                "mcp_server.py"
            ]
        }
    }
)

