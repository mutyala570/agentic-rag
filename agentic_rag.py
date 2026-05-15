from typing import Literal

from langchain_groq import ChatGroq
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, MessagesState, StateGraph
from loguru import logger
from pydantic import BaseModel, Field
from config_loader import get_config
from search_utils import SearchEngine


class GradeDocuments(BaseModel):
    binary_score: str = Field(description="Relevance score: 'yes' if relevant, or 'no' if not relevant")


class AgenticRAG:
    def __init__(
        self,
        model_name: str,
        temperature: float,
        search_engine: SearchEngine,
        grade_prompt: str,
        rewrite_prompt: str,
        generate_prompt: str,
    ):
        self.llm = ChatGroq(model=model_name, temperature=temperature)
        self.grader = ChatGroq(model=model_name, temperature=temperature)
        self.search_engine = search_engine
        self.grade_prompt = grade_prompt
        self.rewrite_prompt = rewrite_prompt
        self.generate_prompt = generate_prompt
        self.graph = self._build_graph()

    def _perform_search(self, state: MessagesState):
        """Always perform search as the first step for any query."""
        last_msg = state["messages"][-1]
        if hasattr(last_msg, 'content'):# return true or false hasattr is a built-in function in Python that checks if an object has a specific attribute. In this case, it checks if the last message in the state has a 'content' attribute. If it does, it retrieves the content directly; otherwise, it tries to get the content using the get method, which is common for dictionaries. This allows the code to handle both cases where messages might be objects with attributes or dictionaries with keys.
            user_query = last_msg.content
        else:
            user_query = last_msg.get("content", "")

        search_results = self.search_engine.hybrid_search(user_query, top_k=10)# search engin for heagvy lefting first node for search
        return {"messages": state["messages"] + [{"role": "assistant", "content": search_results}]}

    def _grade_documents(self, state: MessagesState) -> Literal["generate_answer", "rewrite_question"]:
        """Grade documents and return routing decision directly."""
        first_msg = state["messages"][0]
        if hasattr(first_msg, 'content'):
            user_question = first_msg.content
        else:
            user_question = first_msg.get("content", "")

        last_msg = state["messages"][-1]
        if hasattr(last_msg, 'content'):
            retrieved_context = last_msg.content
        else:
            retrieved_context = last_msg.get("content", "")

        grading_prompt = self.grade_prompt.format(question=user_question, context=retrieved_context)#config.ymal referance from here 
        grading_response = self.grader.with_structured_output(GradeDocuments).invoke(
            [{"role": "user", "content": grading_prompt}]
        )
        relevance_score = grading_response.binary_score

        if relevance_score == "yes":
            return "generate_answer"
        else:
            return "rewrite_question"

    def _rewrite_question(self, state: MessagesState):
        msg_history = state["messages"]

        first_msg = msg_history[0]
        if hasattr(first_msg, 'content'):
            original_question = first_msg.content
        else:
            original_question = first_msg.get("content", "")
        rewrite_prompt = self.rewrite_prompt.format(question=original_question)
        rewritten_response = self.llm.invoke([{"role": "user", "content": rewrite_prompt}])

        return {"messages": [{"role": "user", "content": rewritten_response.content}]}

    def _generate_answer(self, state: MessagesState):
        first_msg = state["messages"][0]
        if hasattr(first_msg, 'content'):
            user_question = first_msg.content
        else:
            user_question = first_msg.get("content", "")

        last_msg = state["messages"][-1]
        if hasattr(last_msg, 'content'):
            retrieved_context = last_msg.content
        else:
            retrieved_context = last_msg.get("content", "")
        answer_prompt = self.generate_prompt.format(question=user_question, context=retrieved_context)
        final_response = self.llm.invoke([{"role": "user", "content": answer_prompt}])

        return {"messages": [final_response]}

    def _build_graph(self):
        workflow = StateGraph(MessagesState)

        # Add all nodes
        workflow.add_node("perform_search", self._perform_search)
        workflow.add_node("rewrite_question", self._rewrite_question)
        workflow.add_node("generate_answer", self._generate_answer)

        # Always start with search - no conditional routing to skip search
        workflow.add_edge(START, "perform_search")

        # After search, always grade the documents and route conditionally
        workflow.add_conditional_edges(
            "perform_search",
            self._grade_documents,
            {
                "generate_answer": "generate_answer",
                "rewrite_question": "rewrite_question",
            },
        )

        # Generate answer leads to end
        workflow.add_edge("generate_answer", END)

        # Rewrite question loops back to search with new question
        workflow.add_edge("rewrite_question", "perform_search")

        return workflow.compile()

    def query(self, user_query: str) -> str:
        try:
            result = self.graph.invoke({"messages": [{"role": "user", "content": user_query}]})

            last_msg = result["messages"][-1]
            if hasattr(last_msg, 'content'):
                response = last_msg.content
            else:
                response = last_msg.get("content", "")

        except GraphRecursionError as e:
            logger.error(f"Graph recursion error: {e}")
            response = "I'm sorry, I can't answer that question."

        return response


def rag_dag(query: str) -> str:
    config = get_config()

    search_engine = SearchEngine(
        chroma_path=config.get("paths.chroma_path"),
        bm25_index_path=config.get("paths.bm25_index"),
        bm25_metadata_path=config.get("paths.bm25_metadata"),
        embedding_model_name=config.get("models.embedding.name"),
        reranker_model_name=config.get("models.reranker.name"),
        reciprocal_rank_k=config.get("search.reciprocal_rank_k")
    )

    rag = AgenticRAG(
        model_name=config.get("rag.agentic.default_model"),
        temperature=config.get("rag.agentic.default_temperature"),
        search_engine=search_engine,
        grade_prompt=config.get("prompts.grade"),
        rewrite_prompt=config.get("prompts.rewrite"),
        generate_prompt=config.get("prompts.generate")
    )
    return rag.query(query)


if __name__ == "__main__":
    config = get_config()

    search_engine = SearchEngine(
        chroma_path=config.get("paths.chroma_path"),
        bm25_index_path=config.get("paths.bm25_index"),
        bm25_metadata_path=config.get("paths.bm25_metadata"),
        embedding_model_name=config.get("models.embedding.name"),
        reranker_model_name=config.get("models.reranker.name"),
        reciprocal_rank_k=config.get("search.reciprocal_rank_k")
    )

    query = "How does the multi-head attention mechanism work in the Transformer?"
    rag = AgenticRAG(
        model_name=config.get("rag.agentic.default_model"),
        temperature=config.get("rag.agentic.default_temperature"),
        search_engine=search_engine,
        grade_prompt=config.get("prompts.grade"),
        rewrite_prompt=config.get("prompts.rewrite"),
        generate_prompt=config.get("prompts.generate")
    )
    response = rag.query(query)
    print(f"Query: {query}")
    print(f"Response: {response}")


    # emmbeding model,bm25 ,model,reeanker,rrf we need to explore 