import os
import yfinance as yf
from langchain.agents import initialize_agent, Tool
from langchain.chat_models import ChatOpenAI
from langchain.tools import tool
from langchain.schema import SystemMessage
from langchain.agents.agent_types import AgentType
from dotenv import load_dotenv
from python_a2a import A2AClient, Message, TextContent, MessageRole

load_dotenv()

a2a_adk_client = A2AClient("http://localhost:5001/a2a")


def get_advice_from_adk(ticker: str) -> str:
    msg = Message(content=TextContent(text=f"Analyze {ticker}"), role=MessageRole.USER)
    response = a2a_adk_client.send_message(msg)
    return (
        response.content.text
        if hasattr(response.content, "text")
        else str(response.content)
    )


advice_tool = Tool.from_function(
    func=get_advice_from_adk,
    name="get_advice_from_adk",
    description="Gets a BUY/HOLD/SELL recommendation from the stock advisor agent.",
)


@tool
def get_stock_summary(ticker: str) -> str:
    """Returns a summary of the current stock performance for the given ticker symbol."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        name = info.get("shortName", ticker)
        current_price = info.get("regularMarketPrice")
        change = info.get("regularMarketChangePercent")
        market_cap = info.get("marketCap")
        high_52w = info.get("fiftyTwoWeekHigh")
        low_52w = info.get("fiftyTwoWeekLow")

        if None in [current_price, change, market_cap, high_52w, low_52w]:
            return f"Sorry, not enough data available for {ticker}."

        trend = "up" if change > 0 else "down" if change < 0 else "flat"
        change_pct = f"{change:.2f}%"

        return (
            f"{name} is currently trading at ${current_price:.2f}, "
            f"which is {change_pct} today. "
            f"The 52-week range is ${low_52w:.2f} - ${high_52w:.2f}. "
            f"Market capitalization is approximately ${market_cap:,}. "
            f"Overall, the stock is trending {trend} today."
        )
    except Exception as e:
        return f"Could not retrieve stock data for {ticker}: {e}"


tools = [get_stock_summary, advice_tool]

llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

system_msg = SystemMessage(
    content="You are a stock reviewer agent. You only provide general summaries of companies, not financial advice."
)

stock_reviewer_agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
    agent_kwargs={"system_message": system_msg},
)

if __name__ == "__main__":
    print(
        "Welcome to Stock Reviewer Agent. Ask me about any stock (e.g., AAPL, TSLA, MSFT):"
    )
    while True:
        query = input("> ")
        if query.lower() in ("exit", "quit"):
            break
        response = stock_reviewer_agent.run(query)
        print(response)
