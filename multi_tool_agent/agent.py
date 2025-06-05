from google.adk.agents import Agent
import yfinance as yf


def get_financial_statements(ticker: str) -> dict:
    """Retrieves the annual financial statements (income statement, balance sheet, cash flow)
    for a given company ticker symbol. Converts any Timestamp columns/indexes to plain strings
    before serializing to dict.

    Args:
        ticker (str): The ticker symbol of the company (e.g., "AAPL", "MSFT", etc.)

    Returns:
        dict: A dictionary with keys:
              - "status": "success" or "error"
              - "financials": nested dict of the income statement (if successful)
              - "balance_sheet": nested dict of the balance sheet (if successful)
              - "cashflow": nested dict of the cash flow statement (if successful)
              - "error_message": error description (only if status == "error")
    """
    ticker = ticker.strip().upper()

    try:
        tk = yf.Ticker(ticker)

        # 1) Fetch the three annual DataFrames
        income_stmt = tk.financials  # DataFrame with Timestamp columns
        balance_sheet = tk.balance_sheet  # DataFrame with Timestamp columns
        cashflow_stmt = tk.cashflow  # DataFrame with Timestamp columns

        # 2) If all three are empty, return error
        if income_stmt.empty and balance_sheet.empty and cashflow_stmt.empty:
            return {
                "status": "error",
                "error_message": f"No financial statements found for ticker '{ticker}'.",
            }

        # 3) Fill NaNs with 0
        income_stmt = income_stmt.fillna(0)
        balance_sheet = balance_sheet.fillna(0)
        cashflow_stmt = cashflow_stmt.fillna(0)

        # 4) Force all column‐labels (and index‐labels) to become plain strings
        def force_str_index_and_columns(df):
            df = df.copy()
            df.columns = df.columns.astype(str)  # Convert Timestamp columns → str
            df.index = df.index.astype(str)  # Convert index labels → str
            return df

        income_stmt = force_str_index_and_columns(income_stmt)
        balance_sheet = force_str_index_and_columns(balance_sheet)
        cashflow_stmt = force_str_index_and_columns(cashflow_stmt)

        # 5) Now it’s safe to call .to_dict()—no more Timestamp keys
        income_dict = income_stmt.to_dict()
        balance_dict = balance_sheet.to_dict()
        cashflow_dict = cashflow_stmt.to_dict()

        return {
            "status": "success",
            "financials": income_dict,
            "balance_sheet": balance_dict,
            "cashflow": cashflow_dict,
        }

    except Exception as e:
        return {"status": "error", "error_message": str(e)}


root_agent = Agent(
    name="stock_advisor_agent",
    model="gemini-2.0-flash",
    description=(
        "Stock-advisor agent that analyzes a company’s financial statements "
        "and recommends buy, hold, or sell."
    ),
    instruction=(
        "You are a stock-advisor agent. "
        "When the user asks about a ticker, call the provided tool "
        "`get_financial_statements(ticker)` to retrieve the latest annual financials. "
        "Based on revenue trends, profit margins, debt levels, and cash flow, "
        "formulate a concise recommendation: BUY, HOLD, or SELL. "
        "Explain your reasoning (e.g., recent revenue growth, debt ratios, profitability trend)."
    ),
    tools=[get_financial_statements],
)
