def chunk_text(text, page_number, document_name, chunk_size=200):

    paragraphs = text.split("\n")

    metadata = []

    current_chunk = ""

    for para in paragraphs:

        para = para.strip()

        if not para:
            continue

        if len(current_chunk) + len(para) < chunk_size:

            current_chunk += para + "\n"

        else:

            metadata.append(
                {
                    "text": current_chunk,
                    "page": page_number,
                    "document": document_name
                }
            )

            current_chunk = para + "\n"

    if current_chunk:

        metadata.append(
            {
                "text": current_chunk,
                "page": page_number,
                "document": document_name
            }
        )

    return metadata