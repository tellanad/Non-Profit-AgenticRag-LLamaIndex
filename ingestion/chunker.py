from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode

from config import settings
from config.logging_config import get_logger

logger = get_logger(__name__)


def chunk_documents(documents: list[Document]) ->list[TextNode]:
    """Split documents into smaller textnode chunks 

        Metadata(filename, page, section)-> will be carried along with each text node
        
        Args:   
            list of documents from parse_pdf
        
        Returns :
            List of TextNodes ready for embeddings 
        
    """
    if not  documents:
        logger.warning('no document found to split')
        return []
    
    #--Create splitter with settings from config --#
    splitter= SentenceSplitter(
        chunk_size= settings.chunk_size,
        chunk_overlap= settings.chunk_overlap
    )
    
    #--Split --#
    nodes = splitter.get_nodes_from_documents(documents)
    
    #--log Useful stats for tuning --#
    avg_len= sum(len(n.text) for n in nodes)// max(len(nodes), 1)
    
    logger.info(
        f"Chunked {len(documents)} documents → {len(nodes)} nodes "
        f"(avg {avg_len} chars/chunk, "
        f"chunk_size={settings.chunk_size}, overlap={settings.chunk_overlap})"
    )
    return nodes