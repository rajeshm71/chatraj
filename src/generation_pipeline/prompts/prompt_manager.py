from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate

class PromptManager:
    def __init__(self):
        """
        Initialize the PromptManager.
        """
        pass  # No initialization needed since methods define individual prompts

    def get_contextualize_question_prompt(self, chat_history_placeholder="chat_history", user_input="{input}"):
        """
        Generate the 'contextualize question' prompt.

        :param chat_history_placeholder: The placeholder for chat history.
        :param user_input: The placeholder for user input.
        :return: ChatPromptTemplate for contextualizing a question.
        """
        return ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "Given a chat history and the latest user question "
                        "which might reference context in the chat history, "
                        "formulate a standalone question which can be understood "
                        "without the chat history. Do NOT answer the question, "
                        "just reformulate it if needed and otherwise return it as is."
                    ),
                ),
                MessagesPlaceholder(chat_history_placeholder),
                ("human", user_input),
            ]
        )

    def get_qa_prompt(self, context_placeholder="{context}", chat_history_placeholder="chat_history", user_input="{input}"):
        """
        Generate the QA prompt.

        :param context_placeholder: The placeholder for context.
        :param chat_history_placeholder: The placeholder for chat history.
        :param user_input: The placeholder for user input.
        :return: ChatPromptTemplate for answering questions with context.
        """
        return ChatPromptTemplate.from_messages(
            [
                ("system", "You are a helpful AI assistant. Use the following context to answer the user's question."),
                ("system", f"Context: {context_placeholder}"),
                MessagesPlaceholder(variable_name=chat_history_placeholder),
                ("human", user_input),
            ]
        )

    def get_context_based_question_answering_prompt(self, context="{context}", question="{question}"):
        """
        Generate the prompt for answering a question based only on the provided context.

        :param context: The placeholder for context.
        :param question: The placeholder for the question.
        :return: ChatPromptTemplate for context-based question answering.
        """
        template = """Answer the question based only on the following context:
                        {context}

                        Question: {question}

                        Answer: """
        return ChatPromptTemplate.from_template(template)

if __name__ == "__main__":
    # Initialize PromptManager
    prompt_manager = PromptManager()

    # Get 'contextualize question' prompt
    contextualize_prompt = prompt_manager.get_contextualize_question_prompt()

    # Get 'QA' prompt
    qa_prompt = prompt_manager.get_qa_prompt(context_placeholder="{custom_context}", user_input="What is AI?")

    # Get 'context-based question answering' prompt
    context_based_prompt = prompt_manager.get_context_based_question_answering_prompt(
        context="The AI system is designed to automate repetitive tasks.",
        question="What is the purpose of the AI system?"
    )

    # print("Contextualize Question Prompt:", contextualize_prompt)
    # print("QA Prompt:", qa_prompt)
    print("qa Prompt:",context_based_prompt.invoke({ "context":"The AI system is designed to automate repetitive tasks.",
        "question":"What is the purpose of the AI system?"}))
    # print("Context-Based Question Answering Prompt:", context_based_prompt)
