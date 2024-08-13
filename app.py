from dotenv import load_dotenv
from typing import List, Sequence
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, MessageGraph
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

class Reflection:
    def __init__(self, task: str, reflection_prompt: str, generation_prompt: str):
        load_dotenv()
        self.task = task

        # Construimos los prompts correctamente usando ChatPromptTemplate
        self.reflection_prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", reflection_prompt),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        self.generation_prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", generation_prompt),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        self.llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")

        # Creamos las cadenas usando los prompts y el LLM
        self.reflect_chain = self.reflection_prompt_template | self.llm
        self.generate_chain = self.generation_prompt_template | self.llm

        

    def build_agent(self):
        def generation_node(state: Sequence[BaseMessage]):
            return self.generate_chain.invoke({"messages": state})

        def reflect_node(state: Sequence[BaseMessage]) -> List[BaseMessage]:
            res = self.reflect_chain.invoke({"messages": state})
            return [HumanMessage(content=res.content)]
        
        REFLECT = "reflect"
        GENERATE = "generate"

        builder = MessageGraph()

        builder.add_node(GENERATE, generation_node)
        builder.add_node(REFLECT, reflect_node)

        builder.set_entry_point(GENERATE)

        def should_continue(state: List[BaseMessage]):
            if len(state) > 3:
                return END
            return REFLECT

        builder.add_conditional_edges(GENERATE, should_continue)
        builder.add_edge(REFLECT, GENERATE)

        graph = builder.compile()

        print(graph.get_graph().draw_mermaid())
        
        inputs = HumanMessage(content=self.task)       
        
        response = graph.invoke(inputs)
        
        return response[-1].content

if __name__ == "__main__":
    reflection_instance = Reflection(
        task="""Make this tweet better:
                          
                          AI is not for everyone.
                          
                          As bitcoins wasnt for everyone in 2012 ;)
                          
                          I made video for you to check out.
                          """,
        generation_prompt="""You are a techie influencer assistant tasked with writing excellent twitter posts.
                              Generate the best twitter post possible for the user's request.
                              If the user provides critique, respond with a reviewed version of your previous attempts
                              """,
        reflection_prompt="""You are a viral twitter influencer grading a tweet. Generate critique and recommendations for the user's tweet.
                              Always provide detailed recommendations including requests for length, virality, style, etc."""
    )
    
    response = reflection_instance.build_agent()
    print(response[-1].content)
