from fastapi import FastAPI
from pydantic import BaseModel
from langchain.chat_models import AzureChatOpenAI
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from urllib.parse import quote_plus
from fastapi.middleware.cors import CORSMiddleware
from langchain.agents import AgentType
from langchain_experimental.sql.base import SQLDatabaseChain
from langchain.callbacks import get_openai_callback
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import sqlalchemy
from typing import List, Dict, Any
from langchain.prompts import PromptTemplate
import json


# Set up Azure OpenAI Configuration
azure_deployment_name = "gpt-4o"  # Replace with your Azure GPT-4 deployment name
azure_openai_api_key = "9nh861Kj4U5vBzSqHCfbwJ1EloO8rhXOKoVt9WE8qqIH0Wcd5YwZJQQJ99ALACHYHv6XJ3w3AAAAACOGubrO"
azure_openai_base_url = "https://stackyon-ai-services.openai.azure.com/"  # Replace with your Azure OpenAI URL

# Correct Azure OpenAI Initialization for LangChain
langchain = AzureChatOpenAI(
    deployment_name=azure_deployment_name,
    azure_endpoint=f"{azure_openai_base_url}",
    openai_api_key=azure_openai_api_key,
    openai_api_type="azure",
    openai_api_version="2023-05-15",
    temperature=0,
    max_tokens=5000
)

# Configure MySQL Database Connection
mysql_database_name = "ifb_db"
mysql_user = "admin"
mysql_password = quote_plus("KEgIWoE3aHTESn")  # Use quote_plus to encode special characters
mysql_host = "stkqadb.c6p27q3bgq8w.us-east-1.rds.amazonaws.com"
mysql_port = 3306

# Create SQLDatabase connection
sql_database = SQLDatabase.from_uri(
    f"mysql+pymysql://{mysql_user}:{mysql_password}@{mysql_host}:{mysql_port}/{mysql_database_name}"
)


# Create SQLDatabaseToolkit
toolkit = SQLDatabaseToolkit(db=sql_database, llm=langchain)

sql_db_chain = SQLDatabaseChain.from_llm(llm = langchain, db = sql_database, verbose = True)

prompt_template = """You are a SQL expert.
 Use the tools available to answer questions about the database. There are audit columns in every table (like __region_id,__created_by,__last_updated_by,__created_by_name,__last_updated_by_name,__created_on,__last_updated_on,__is_delete,__event_settings,BatchId) dont use in Select query.
 Use Unique regions from tblregion table and Isdefault=1.Any table consider only __is_delete=0 except tblregion which doesn't __is_delete column.
 Parent region is India. Under India , Four Child regions are there.North (North India),West (West India),South (South India), East (East India),Central regions.Under Every Child region, there are branches (regions).if quesion is region Wise then (Sum of their child branches).
 Employee table contains employee Details and empcategory as Employee category ( Consider Only CAPABLE,TRAINER,XP,MASTER,EXPERT and remaining consider as OTHER) And Status as Active or Inactive.
**IMPORTANT**:
1. Always use MySQL-compatible SQL syntax.
2. Wrap column names in backticks (`) if they contain special characters.
3. Do not generate queries that reference non-existent tables or columns.
4. Avoid unnecessary joins that might cause performance issues.

Example Query:
**Input:** "Show top 5 regions "
**Output SQL:** `SELECT distinct RegionName FROM tblregion where IsDefault=1 LIMIT 5;`

Available tools:
{tools}

Use the following format:
Question: the input question you must answer
Thought: what you think you should do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
...
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
{agent_scratchpad}
"""

promptt = PromptTemplate(
    input_variables=["input", "tool_names", "tools", "agent_scratchpad"],
    template=prompt_template,
)

# Initialize the SQL agent
sql_agent = create_sql_agent(
    llm=langchain,
    toolkit=toolkit,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    agent_executor_kwargs = {"return_intermediate_steps": True,"handle_parsing_errors": True},
    prompt=promptt
)

# Define a function to handle NLP queries and return table rows
# def process_nlp_query(query):
#     try:
#         print(f"Processing query: {query}")
#         # Check if SQL agent is configured
#         if not sql_agent:
#             raise ValueError("SQL Agent is not properly initialized.")
        
#         # Run the query using LangChain's SQL agent
#         response = sql_agent.invoke(query)
#         #print(response)
#         generated_sql = "none"
#         tool_input="dd"
#         intermediate_steps = response.get('intermediate_steps', [])
        
#         # Extract the last SQL query from intermediate steps
#         if intermediate_steps:
#             last_step = intermediate_steps[-1]
#             first_action = last_step[0] 
         
#             tool_input = getattr(first_action, "tool_input", "none") 
#             generated_sql = tool_input

#        # generated_sql = last_step.get('tool_input', None)
#        # response_query = pd.read_sql_query(query)
#         # Assuming response is returned as rows, otherwise, adapt as needed
#         if isinstance(response, str):
#             response = {"data": response}  # Fallback for string-based responses
        
#         return {"success": first_action, "data": tool_input,"query":generated_sql}
#     except Exception as e:
#         error_details = f"Error while processing query: {str(e)}"
#        # print(error_details)
#         return {"success": False, "error": error_details}


# def process_nlp_query(query: str,type: str) -> dict:
#     """
#     Process an NLP query using LangChain's SQL agent.

#     Args:
#         query (str): The NLP query to be processed.

#     Returns:
#         dict: A dictionary containing success status, data, and generated SQL query or error details.
#     """
#     try:
#         # Validate input
#         if not query:
#             raise ValueError("Query cannot be empty.")

#         print(f"Processing query: {query}")
        
#         # Check if SQL agent is configured
#         if not sql_agent:
#             raise ValueError("SQL Agent is not properly initialized.")
        
#         # Run the query using LangChain's SQL agent
#         #response = sql_agent.invoke(query)
#         response = sql_agent.invoke({"input": query}, config={"handle_parsing_errors": True})
#         parsed_output = json.loads(response)
#         # Ensure response is valid
#         if not isinstance(parsed_output, dict):
#             raise ValueError("Unexpected response format from SQL agent.")
        
#         # Extract the intermediate steps
#         intermediate_steps = parsed_output.get('intermediate_steps', [])
#         generated_sql = "none"
#         tool_input = "none"

#         if intermediate_steps:
#             last_step = intermediate_steps[-1]
#             first_action = last_step[0] if last_step else None
            
#             if first_action:
#                 tool_input = getattr(first_action, "tool_input", "none")
#                 generated_sql = tool_input


#         # Session = sessionmaker(bind=engine)
#         # with Session() as session:
#         #     result = session.execute(generated_sql)
            
#         #     # Fetch and format results
#         #     columns = result.keys()
#         #     data = [dict(zip(columns, row)) for row in result]
#         data=[]
#         if type == "report":
#             engine = sqlalchemy.create_engine(
#                 f"mysql+pymysql://{mysql_user}:{mysql_password}@{mysql_host}:{mysql_port}/{mysql_database_name}"
#             )
        
#             with engine.connect() as connection:
#                 # Execute the query
#                 result = connection.execute(sqlalchemy.text(generated_sql))
                
#                 # Get column names
#                 columns = result.keys()
                
#                 # Fetch all rows
#                 rows = result.fetchall()
                
#                 # Convert rows to list of dictionaries
#                 data = [dict(zip(columns, row)) for row in rows]

#         return {
#             "success": True,
#             "input": parsed_output.get('input', ''),
#             "output": parsed_output.get('output', ''),
#             "query": generated_sql,
#             "data": data,
#         }

#     except ValueError as ve:
#         error_details = f"Validation error: {str(ve)}"
#         return {"success": False, "error": error_details}
    
#     except AttributeError as ae:
#         error_details = f"Attribute error: {str(ae)}"
#         return {"success": False, "error": error_details}
    
#     except Exception as e:
#         error_details = f"Unexpected error: {str(e)}"
#         return {"success": False, "error": error_details}
def is_safe_sql(query: str) -> bool:
    query = query.strip().lower()
    return query.startswith("select")
def process_nlp_query(query: str, type: str) -> dict:
    """
    Process an NLP query using LangChain's SQL agent.

    Args:
        query (str): The NLP query to be processed.
        type (str): The type of query processing ("report" or other types).

    Returns:
        dict: A dictionary containing success status, data, and generated SQL query or error details.
    """
    try:
        if not query:
            raise ValueError("Query cannot be empty.")

        print(f"Processing query: {query}")
        
        if not sql_agent:
            raise ValueError("SQL Agent is not properly initialized.")
        
        response = sql_agent.invoke({"input": query}, config={"handle_parsing_errors": True})

        parsed_output = " "

        if isinstance(response, dict):
            parsed_output = response
        else:
            parsed_output = json.loads(response)
        data = []
     

        intermediate_steps = parsed_output.get('intermediate_steps', [])
        generated_sql = "none"

        if intermediate_steps:
            last_step = intermediate_steps[-1]
            first_action = last_step[0] if last_step else None
            
            if first_action:
                tool_input = getattr(first_action, "tool_input", "none")
                generated_sql = tool_input


            if not generated_sql or not is_safe_sql(generated_sql):
                return {"success": False, "error": "Generated query is invalid or unsafe.","input": generated_sql,}

        print("Generated SQL:", generated_sql)
        # ✅ Validate SQL before execution
        

        engine = sqlalchemy.create_engine(
                f"mysql+pymysql://{mysql_user}:{mysql_password}@{mysql_host}:{mysql_port}/{mysql_database_name}"
            )

        with engine.connect() as connection:
                try:
                    result = connection.execute(sqlalchemy.text(generated_sql))
                    columns = result.keys()
                    rows = result.fetchall()
                    data = [dict(zip(columns, row)) for row in rows]
                except sqlalchemy.exc.SQLAlchemyError as db_error:
                    return {"success": False, "error": f"Database error: {str(db_error)}"}
        
        return {
            "success": True,
            "input": parsed_output.get('input', ''),
            "output": parsed_output.get('output', ''),
            "query": generated_sql,
            "data": data,
        }

    except ValueError as ve:
        return {"success": False,  "input": " ",
            "output": " ", "Validation error": f"Validation error: {str(ve)}"}
    
    except AttributeError as ae:
        return {"success": False,  "input": " ",
            "output": " ", "Attribute error": f"Attribute error: {str(ae)}"}
    
    except Exception as e:
        return {"success": False,  "input": " ",
            "output": " ", "Unexpected error": f"Unexpected error: {str(e)}"}



def process_nlp_queryChat(query: str, type: str) -> dict:
    """
    Process an NLP query using LangChain's SQL agent.

    Args:
        query (str): The NLP query to be processed.
        type (str): The type of query processing ("report" or other types).

    Returns:
        dict: A dictionary containing success status, data, and generated SQL query or error details.
    """
    try:
        if not query:
            raise ValueError("Query cannot be empty.")

        print(f"Processing query: {query}")
        
        if not sql_agent:
            raise ValueError("SQL Agent is not properly initialized.")
        
        response = sql_agent.invoke({"input": query}, config={"handle_parsing_errors": True})

        parsed_output = " "

        if isinstance(response, dict):
            parsed_output = response
        else:
            parsed_output = json.loads(response)
        
        
        return {
            "success": True,
            "input": parsed_output.get('input', ''),
            "output": parsed_output.get('output', ''),
           
        }

    except ValueError as ve:
        return {"success": False,  "input": " ",
            "output": " ", "Validation error": f"Validation error: {str(ve)}"}
    
    except AttributeError as ae:
        return {"success": False,  "input": " ",
            "output": " ", "Attribute error": f"Attribute error: {str(ae)}"}
    
    except Exception as e:
        return {"success": False,  "input": " ",
            "output": " ", "Unexpected error": f"Unexpected error: {str(e)}"}









# FastAPI App
app = FastAPI()

# CORS Configuration
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allow specific origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Define root endpoint
@app.get("/")
def root():
    return {"message": "Hello World"}

# Define a Pydantic model for query payload
class QueryPayload(BaseModel):
    query: str
    querytype:str

# POST endpoint to process query
@app.post("/talktoDB")
def queryDB(payload: QueryPayload):
    request = payload.query
    type = payload.querytype
    print("Processing query:", request)
    result = process_nlp_queryChat(request,"chat")
    return result
            
       
@app.post("/reportDB")
def queryDB(payload: QueryPayload):
    request = payload.query
    type = payload.querytype
    print("Processing query:", request)
    result = process_nlp_query(request,"report")
    return result
                    
          

    #print("Processing query:", request)
    #result = process_nlp_query(request)
    return result

# GET endpoint to process query
@app.get("/talktosql")
def queryDB(request: str):
    print("Processing query:", request)
    result = process_nlp_query(request,"report")
    return result
