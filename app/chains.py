import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from dotenv import load_dotenv

load_dotenv()

class Chain:
    def __init__(self):
        self.llm = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.1,
            max_tokens=1000,
            model_name="llama-3.3-70b-versatile")
    
    def extracted_jobs(self,cleaned_text):
        prompt_extract=PromptTemplate.from_template(
            """
            ###SCRAPED TEXT FROM WEBSITE:
            {page_data}
            #INSTRUCTION:
            The scraped text is from the career's page of a website,
            Your job is to extract the job postings and return them in json format containing the following keys:
            'roles','exerience','skills' and 'description'.
            Only return the valid json.
            #VALID JSON(NO PREAMBLE):
            """
        )
        chain_extract=prompt_extract | self.llm
        response=chain_extract.invoke(input={'page_data': cleaned_text})
        try:
            json_parser = JsonOutputParser()
            response = json_parser.parse(response.content)
        except OutputParserException:
            raise OutputParserException("Context too big. Unable to parse jobs.")
        return response if isinstance(response, list) else [response]
    
    def write_email(self, job, links):
        prompt_email = PromptTemplate.from_template(
            """
            ### JOB DESCRIPTION:
            {job_description}
            
            ### INSTRUCTION:
            You are Raj, a computer science engineer with 5 years of experience.  
            Your job is to write a cold email to the client regarding the job mentioned above describing the capability of Raj 
            in fulfilling his needs.
            Also add the most relevant ones from the following links to showcase Raj's portfolio: {link_list}
            Remember you are Raj, a computer science engineer. 
            Do not provide a preamble.
            ### EMAIL (NO PREAMBLE):
            
            """
        )

        chain_email = prompt_email | self.llm
        response = chain_email.invoke({"job_description": str(job), "link_list": links})
        return response.content