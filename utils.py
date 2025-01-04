def docs2str(docs):
    return "\n\n".join(doc.page_content for doc in docs)