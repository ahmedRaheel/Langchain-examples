from datetime import datetime
from fastmcp import FastMCP
import random

mcp = FastMCP("Time server")

@mcp.tool

def get_current_time():
    """Return the current date and time."""

    return datetime.now().strftime("%d-%m-%Y %I:%M:%S %p")

@mcp.tool
def roll_dice():
    """Return a random number between 1 and 6."""

    return random.randint(1, 6)

if __name__ == "__main__" :
    mcp.run()
