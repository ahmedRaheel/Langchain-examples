import os
import requests
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from langchain.tools import tool
from langchain_ollama import ChatOllama
from langchain.agents import create_agent

from dotenv import load_dotenv

load_dotenv()


# -----------------------------
# Tools
# -----------------------------

@tool
def get_weather_forecast(city: str) -> str:
    """Get current weather forecast for a city."""
    try:
        response = requests.get(
            "https://api.openweathermap.org/data/2.5/weather",
            params={
                "q": city,
                "appid": os.getenv("OPENWEATHER_API_KEY"),
                "units": "metric"
            }
        )

        data        = response.json()
        temp        = data["main"]["temp"]
        feels_like  = data["main"]["feels_like"]
        description = data["weather"][0]["description"]
        humidity    = data["main"]["humidity"]

        return (
            f"Weather in {city}:\n"
            f"  Temperature : {temp}°C (feels like {feels_like}°C)\n"
            f"  Condition   : {description}\n"
            f"  Humidity    : {humidity}%"
        )

    except Exception as e:
        return f"Weather fetch error: {e}"


@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email to a recipient."""
    try:
        smtp_host       = "smtp.gmail.com"
        smtp_port       = 587
        sender_email    = os.getenv("EMAIL_ADDRESS")
        sender_password = os.getenv("EMAIL_PASSWORD")

        msg            = MIMEMultipart()
        msg["From"]    = sender_email
        msg["To"]      = to
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.ehlo()
            server.starttls()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, to, msg.as_string())

        return f"Email sent to {to}"

    except Exception as e:
        return f"Email error: {e}"


# -----------------------------
# Model
# -----------------------------

model = ChatOllama(
    model="qwen2.5:0.5b",   # or "mistral", "qwen2.5", etc.
    temperature=0
)


# -----------------------------
# Agent
# -----------------------------

tools = [
    get_weather_forecast,
    send_email
]

agent = create_agent(
    model=model,
    tools=tools,
    system_prompt="""
You are a helpful assistant.

Rules:
1. Use get_weather_forecast to get weather for a city.
2. Use send_email to send emails to recipients.
3. Always confirm after sending an email.
4. Provide a concise final answer after completing tool calls.
"""
)


# -----------------------------
# Execute Agent
# -----------------------------


result = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": "What is the weather in Karachi? Then send an email to raaheelahmed@outlook.com with the weather update."
        }
    ]
})

print(result["messages"][-1].content)