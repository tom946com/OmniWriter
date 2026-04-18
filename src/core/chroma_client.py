import os
from typing import Any, List, Dict, Optional
import chromadb
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from src.utils.logs import logger

load_dotenv()


class ChromaClient:
    """
    ChromaDB客户端封装类，提供文档嵌入、存储和查询功能
    """
    
    def __init__(self, 
                 collection_name: str = "default_collection",
                 embedding_model: Optional[str] = None):
        """
        初始化Chroma客户端
        
        :param collection_name: 集合名称
        :param embedding_model: 嵌入模型名称，如果为None则从环境变量读取
        """
        self.collection_name = collection_name
        
        # 创建Chroma客户端
        storage_path = os.getenv("CHROMA_STORAGE_PATH", "./data/chroma")
        os.makedirs(storage_path, exist_ok=True)
        self.client = chromadb.PersistentClient(path=storage_path)
        
        # 初始化嵌入函数
        self._embeddings = self._create_embedding_client(embedding_model)
        
        # 获取或创建集合
        self.collection = self._get_or_create_collection()
    
    def _create_embedding_client(self, embedding_model: Optional[str] = None) -> OpenAIEmbeddings:
        """
        创建嵌入向量客户端
        
        :param embedding_model: 嵌入模型名称
        :return: OpenAIEmbeddings 实例
        """
        base_url = os.getenv("EMBEDDING_BASE_URL")
        api_key = os.getenv("EMBEDDING_BASE_API_KEY")
        model = embedding_model or os.getenv("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-4B")
        
        embeddings = OpenAIEmbeddings(
            base_url=base_url,
            api_key=api_key,
            model=model,
            dimensions=1024
        )
        return embeddings
    
    def _get_or_create_collection(self):
        """
        获取或创建集合
        
        :return: Chroma 集合对象
        """
        try:
            collection = self.client.get_collection(name=self.collection_name)
            return collection
        except Exception:
            collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            return collection
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        生成文本的嵌入向量
        
        :param texts: 文本列表
        :return: 嵌入向量列表
        """
        return self._embeddings.embed_documents(texts)
    
    def get_embedding(self, text: str) -> List[float]:
        """
        生成单个文本的嵌入向量
        
        :param text: 文本
        :return: 嵌入向量
        """
        return self._embeddings.embed_query(text)
    
    def add_documents(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ) -> None:
        """
        添加文档到集合（使用自定义嵌入向量）
        
        :param texts: 文本内容列表
        :param metadatas: 元数据列表
        :param ids: 文档ID列表，不提供则自动生成
        """
        if not texts:
            logger.warning("No documents to add")
            return
        
        # 生成嵌入向量
        embeddings = self.get_embeddings(texts)
        
        # 添加文档
        self.collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
    
    def add_document(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        doc_id: Optional[str] = None,
    ) -> None:
        """
        添加单个文档
        
        :param text: 文本内容
        :param metadata: 元数据
        :param doc_id: 文档ID
        """
        self.add_documents([text], [metadata] if metadata else None, [doc_id] if doc_id else None)
    
    def query(
        self,
        query_texts: Optional[List[str]] = None,
        query_embeddings: Optional[List[List[float]]] = None,
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, Any]] = None,
        include: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        查询相似文档
        
        :param query_texts: 查询文本列表（如果提供了query_embeddings则不需要）
        :param query_embeddings: 查询嵌入向量列表（如果提供了query_texts则不需要）
        :param n_results: 返回结果数量
        :param where: 元数据过滤条件
        :param where_document: 文档内容过滤条件
        :param include: 包含的字段，默认["documents", "metadatas", "distances"]
        :return: 查询结果
        """
        if include is None:
            include = ["documents", "metadatas", "distances"]
        
        if query_texts and not query_embeddings:
            query_embeddings = self.get_embeddings(query_texts)
        
        if not query_embeddings:
            raise ValueError("Either query_texts or query_embeddings must be provided")
        
        results = self.collection.query(
            query_embeddings=query_embeddings,
            n_results=n_results,
            where=where,
            where_document=where_document,
            include=include
        )
        
        return results
    
    def query_by_text(
        self,
        text: str,
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        通过文本查询相似文档
        
        :param text: 查询文本
        :param n_results: 返回结果数量
        :param where: 元数据过滤条件
        :return: 查询结果
        """
        return self.query(query_texts=[text], n_results=n_results, where=where)
    
    def get(
        self,
        ids: Optional[List[str]] = None,
        where: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        where_document: Optional[Dict[str, Any]] = None,
        include: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        获取文档
        
        :param ids: 文档ID列表
        :param where: 元数据过滤条件
        :param limit: 返回数量限制
        :param offset: 偏移量
        :param where_document: 文档内容过滤条件
        :param include: 包含的字段
        :return: 文档数据
        """
        if include is None:
            include = ["documents", "metadatas"]
        
        results = self.collection.get(
            ids=ids,
            where=where,
            limit=limit,
            offset=offset,
            where_document=where_document,
            include=include
        )
        return results
    
    def update_documents(
        self,
        ids: List[str],
        texts: Optional[List[str]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """
        更新文档
        
        :param ids: 文档ID列表
        :param texts: 新文本内容列表
        :param metadatas: 新元数据列表
        """
        if texts:
            embeddings = self.get_embeddings(texts)
            self.collection.update(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas
            )
        else:
            self.collection.update(
                ids=ids,
                metadatas=metadatas
            )
    
    def delete(
        self,
        ids: Optional[List[str]] = None,
        where: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        删除文档
        
        :param ids: 文档ID列表
        :param where: 元数据过滤条件
        :param where_document: 文档内容过滤条件
        """
        self.collection.delete(
            ids=ids,
            where=where,
            where_document=where_document
        )
        logger.info("Documents deleted")
    
    def count(self) -> int:
        """
        获取集合中文档数量
        
        :return: 文档数量
        """
        return self.collection.count()
    
    def peek(self, limit: int = 10) -> Dict[str, Any]:
        """
        查看集合中的前N个文档
        
        :param limit: 返回数量
        :return: 文档数据
        """
        return self.collection.peek(limit)
    
    def del_collection(self) -> None:
        """
        删除整个集合
        """
        self.client.delete_collection(name=self.collection_name)
        # 重新创建空集合
        self.collection = self._get_or_create_collection()
    
    def list_collections(self) -> List[str]:
        """
        列出所有集合名称
        
        :return: 集合名称列表
        """
        collections = self.client.list_collections()
        return [col.name for col in collections]


chroma_client = ChromaClient()
# print(chroma_client.list_collections())
# chroma_client.client.delete_collection(name="default_collection")
# print(chroma_client.list_collections()) 
