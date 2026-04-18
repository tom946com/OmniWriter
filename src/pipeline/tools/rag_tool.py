"""
RAG 工具类
提供网页数据分块、同时存入 ES 和 Chroma 数据库，以及混合搜索功能
"""

from typing import List, Dict, Any, Optional
from collections import defaultdict
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.core.chroma_client import ChromaClient
from src.core.es_client import ESClient
from src.utils.logs import logger


class RAGTool:
    """
    RAG 工具类
    整合 ES 全文搜索和 Chroma 向量搜索，提供统一的存储和查询接口
    """

    def __init__(
        self,
        collection_name: Optional[str] = None,
        index_name: Optional[str] = None
    ):
        """
        初始化 RAG 工具

        Args:
            collection_name: Chroma 集合名称
            index_name: ES 索引名称
        """
        self.collection_name = collection_name or "default_collection"
        self.index_name = index_name or "default_index"

        self.chroma_client = ChromaClient(collection_name=self.collection_name)
        self.es_client = ESClient()

        self.default_chunk_size = 512
        self.default_chunk_overlap = 50

        self._ensure_es_index_exists()

    def _ensure_es_index_exists(self):
        """确保 ES 索引存在"""
        if not self.es_client.index_exists(self.index_name):
            mappings = {
                "properties": {
                    "title": {"type": "text", "analyzer": "icu_analyzer"},
                    "content": {"type": "text", "analyzer": "icu_analyzer"},
                    "url": {"type": "keyword"},
                    "chunk_index": {"type": "integer"},
                    "total_chunks": {"type": "integer"},
                    "task_id": {"type": "keyword"},
                    "source": {"type": "keyword"},
                    "created_at": {"type": "date"}
                }
            }
            settings = {
                "number_of_shards": 1,
                "number_of_replicas": 0
            }
            self.es_client.create_index(
                index_name=self.index_name,
                mappings=mappings,
                settings=settings
            )

    def split_text_into_chunks(
        self,
        text: str,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None
    ) -> List[str]:
        """
        使用 LangChain 的 RecursiveCharacterTextSplitter 进行文本分块
        针对中文进行了专门优化

        Args:
            text: 待分块的文本
            chunk_size: 分块大小（字符数）
            chunk_overlap: 分块重叠（字符数）

        Returns:
            分块后的文本列表
        """
        if not text or not text.strip():
            return []

        chunk_size = chunk_size or self.default_chunk_size
        chunk_overlap = chunk_overlap or self.default_chunk_overlap

        separators = [
            "\n\n",
            "\n",
            "。",
            "！",
            "？",
            "；",
            "：",
            "，",
            ". ",
            "! ",
            "? ",
            " ",
            "",
        ]

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            length_function=len,
            is_separator_regex=False,
        )

        chunks = text_splitter.split_text(text)
        chunks = [chunk for chunk in chunks if chunk.strip()]

        return chunks

    def add_webpage(
        self,
        title: str,
        url: str,
        content: str,
        task_id: Optional[str] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None
    ) -> Dict[str, int]:
        """
        将单个网页内容分块并同时添加到 Chroma 和 ES

        Args:
            title: 网页标题
            url: 网页 URL
            content: 网页内容
            task_id: 任务 ID（可选）
            chunk_size: 分块大小
            chunk_overlap: 分块重叠

        Returns:
            添加的文档统计信息
        """
        if not content or not content.strip():
            logger.warning(f"网页内容为空，跳过: {title}")
            return {"chroma_chunks": 0, "es_chunks": 0}

        try:
            chunks = self.split_text_into_chunks(content, chunk_size, chunk_overlap)

            if not chunks:
                logger.warning(f"文本分块结果为空，跳过: {title}")
                return {"chroma_chunks": 0, "es_chunks": 0}

            chroma_texts = []
            chroma_metadatas = []
            chroma_ids = []
            es_documents = []

            for i, chunk in enumerate(chunks):
                chunk_id = f"{task_id or 'default'}_{url.replace('https://', '').replace('http://', '').replace('/', '_')[:50]}_{i}"

                metadata = {
                    "title": title,
                    "url": url,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "source": "webpage"
                }
                if task_id:
                    metadata["task_id"] = task_id

                chroma_texts.append(chunk)
                chroma_metadatas.append(metadata)
                chroma_ids.append(chunk_id)

                es_doc = {
                    "title": title,
                    "content": chunk,
                    "url": url,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "source": "webpage"
                }
                if task_id:
                    es_doc["task_id"] = task_id
                es_documents.append((chunk_id, es_doc))

            self.chroma_client.add_documents(
                texts=chroma_texts,
                metadatas=chroma_metadatas,
                ids=chroma_ids
            )

            for doc_id, doc in es_documents:
                self.es_client.upsert_document(
                    index_name=self.index_name,
                    document=doc,
                    doc_id=doc_id
                )

            logger.info(f"网页内容已添加: {title}, {len(chunks)} 块")
            return {"chroma_chunks": len(chunks), "es_chunks": len(chunks)}

        except Exception as e:
            logger.error(f"添加网页失败: {title}, 错误: {str(e)}", exc_info=True)
            raise e
            return {"chroma_chunks": 0, "es_chunks": 0}

    def add_search_results(
        self,
        search_results: List[Dict[str, Any]],
        task_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        批量将搜索结果添加到 Chroma 和 ES

        Args:
            search_results: 搜索结果列表，每个结果包含 title, url, content
            task_id: 任务 ID（可选）

        Returns:
            统计信息
        """
        total_webpages = 0
        total_chroma_chunks = 0
        total_es_chunks = 0
        failed_webpages = 0

        for result in search_results:
            try:
                title = result.get("title", "")
                url = result.get("url", "")
                content = result.get("content", "")

                if not title or not url:
                    failed_webpages += 1
                    continue

                stats = self.add_webpage(
                    title=title,
                    url=url,
                    content=content,
                    task_id=task_id
                )

                if stats["chroma_chunks"] > 0 or stats["es_chunks"] > 0:
                    total_webpages += 1
                    total_chroma_chunks += stats["chroma_chunks"]
                    total_es_chunks += stats["es_chunks"]
                else:
                    failed_webpages += 1

            except Exception as e:
                failed_webpages += 1

        return {
            "total_webpages_processed": len(search_results),
            "successful_webpages": total_webpages,
            "failed_webpages": failed_webpages,
            "total_chroma_chunks": total_chroma_chunks,
            "total_es_chunks": total_es_chunks
        }

    def search_es(
        self,
        query: str,
        size: int = 5,
        task_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        在 ES 中进行全文检索

        Args:
            query: 查询文本
            size: 返回结果数量
            task_id: 任务 ID 过滤（可选）

        Returns:
            查询结果列表
        """
        try:
            must_conditions = [
                {"match": {"content": query}}
            ]

            if task_id:
                must_conditions.append({"term": {"task_id": task_id}})

            bool_query = {"must": must_conditions}
            response = self.es_client.search(
                index_name=self.index_name,
                query={"bool": bool_query},
                size=size
            )

            hits = response.get("hits", {}).get("hits", [])
            results = []

            for hit in hits:
                results.append({
                    "id": hit.get("_id"),
                    "document": hit.get("_source", {}).get("content", ""),
                    "metadata": hit.get("_source", {}),
                    "score": hit.get("_score"),
                    "source": "es"
                })

            return results

        except Exception as e:
            logger.error(f"ES 搜索失败: {str(e)}", exc_info=True)
            return []

    def search_chroma(
        self,
        query: str,
        n_results: int = 5,
        task_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        在 Chroma 中进行向量检索

        Args:
            query: 查询文本
            n_results: 返回结果数量
            task_id: 任务 ID 过滤（可选）

        Returns:
            查询结果列表
        """
        try:
            where = {"task_id": task_id} if task_id else None
            results = self.chroma_client.query_by_text(
                text=query,
                n_results=n_results,
                where=where
            )

            return self._format_chroma_results(results)

        except Exception as e:
            logger.error(f"Chroma 搜索失败: {str(e)}", exc_info=True)
            return []

    def _format_chroma_results(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        格式化 Chroma 查询结果

        Args:
            results: ChromaDB 原始查询结果

        Returns:
            格式化后的结果列表
        """
        formatted = []

        try:
            ids = results.get("ids", [[]])[0]
            documents = results.get("documents", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0]

            for i in range(len(ids)):
                item = {
                    "id": ids[i] if i < len(ids) else None,
                    "document": documents[i] if i < len(documents) else "",
                    "metadata": metadatas[i] if i < len(metadatas) else {},
                    "distance": distances[i] if i < len(distances) else None,
                    "source": "chroma"
                }
                formatted.append(item)
        except Exception as e:
            logger.error(f"格式化 Chroma 结果失败: {str(e)}", exc_info=True)

        return formatted

    def hybrid_search(
        self,
        query: str,
        es_size: int = 5,
        chroma_size: int = 5,
        task_id: Optional[str] = None,
        deduplicate: bool = True
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        混合搜索：同时从 ES 和 Chroma 中检索结果

        Args:
            query: 查询文本
            es_size: ES 返回结果数量
            chroma_size: Chroma 返回结果数量
            task_id: 任务 ID 过滤（可选）
            deduplicate: 是否去重

        Returns:
            包含 es_results 和 chroma_results 的字典
        """
        es_results = self.search_es(query, size=es_size, task_id=task_id)
        chroma_results = self.search_chroma(query, n_results=chroma_size, task_id=task_id)

        if deduplicate:
            seen_documents = set()
            unique_es = []
            unique_chroma = []

            for res in es_results:
                doc = res.get("document", "")
                if doc not in seen_documents:
                    seen_documents.add(doc)
                    unique_es.append(res)

            for res in chroma_results:
                doc = res.get("document", "")
                if doc not in seen_documents:
                    seen_documents.add(doc)
                    unique_chroma.append(res)

            es_results = unique_es
            chroma_results = unique_chroma

        return {
            "es_results": es_results,
            "chroma_results": chroma_results
        }

    def rrf_fusion(
        self,
        es_results: List[Dict[str, Any]],
        chroma_results: List[Dict[str, Any]],
        k: int = 60
    ) -> List[Dict[str, Any]]:
        """
        使用 RRF（Reciprocal Rank Fusion）算法融合 ES 和 Chroma 的搜索结果
        
        RRF 公式：score = Σ 1 / (k + rank)
        
        Args:
            es_results: ES 搜索结果列表
            chroma_results: Chroma 搜索结果列表
            k: RRF 算法的常数参数，默认为 60
            
        Returns:
            融合并排序后的结果列表
        """
        # 用于存储每个文档的分数
        doc_scores = defaultdict(float)
        # 用于存储每个文档的完整信息
        doc_info = {}
        
        # 处理 ES 结果
        for rank, result in enumerate(es_results, start=1):
            doc_id = result.get("id")
            if not doc_id:
                continue
            # 计算 RRF 分数
            doc_scores[doc_id] += 1.0 / (k + rank)
            # 保存文档信息
            doc_info[doc_id] = result
        
        # 处理 Chroma 结果
        for rank, result in enumerate(chroma_results, start=1):
            doc_id = result.get("id")
            if not doc_id:
                continue
            # 计算 RRF 分数
            doc_scores[doc_id] += 1.0 / (k + rank)
            # 如果文档信息不存在，则保存
            if doc_id not in doc_info:
                doc_info[doc_id] = result
        
        # 按分数降序排序
        sorted_docs = sorted(
            doc_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # 构建最终结果列表
        final_results = []
        for doc_id, score in sorted_docs:
            result = doc_info[doc_id].copy()
            result["rrf_score"] = score
            final_results.append(result)
        
        logger.info(f"RRF 融合完成，共 {len(final_results)} 个结果")
        return final_results

    def hybrid_search_rrf(
        self,
        query: str,
        es_size: int = 10,
        chroma_size: int = 10,
        top_k: int = 10,
        task_id: Optional[str] = None,
        k: int = 60
    ) -> List[Dict[str, Any]]:
        """
        混合搜索并使用 RRF 算法融合结果
        
        Args:
            query: 查询文本
            es_size: ES 返回结果数量
            chroma_size: Chroma 返回结果数量
            top_k: 返回的最终结果数量
            task_id: 任务 ID 过滤（可选）
            k: RRF 算法的常数参数
            
        Returns:
            融合排序后的 top_k 个结果
        """
        # 分别从 ES 和 Chroma 检索
        hybrid_result = self.hybrid_search(
            query=query,
            es_size=es_size,
            chroma_size=chroma_size,
            task_id=task_id,
            deduplicate=False
        )
        
        # 使用 RRF 融合
        fused_results = self.rrf_fusion(
            es_results=hybrid_result["es_results"],
            chroma_results=hybrid_result["chroma_results"],
            k=k
        )
        
        # 返回 top_k 个结果
        return fused_results[:top_k]

    def get_all_task_materials(
        self,
        task_id: str,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        获取指定 task_id 的所有素材
        
        Args:
            task_id: 任务 ID
            limit: 返回结果数量限制（可选）
            
        Returns:
            素材列表
        """
        try:
            # 从 Chroma 获取
            where = {"task_id": task_id}
            chroma_results = self.chroma_client.get(
                where=where,
                limit=limit
            )
            
            materials = []
            ids = chroma_results.get("ids", [])
            documents = chroma_results.get("documents", [])
            metadatas = chroma_results.get("metadatas", [])
            
            for i in range(len(ids)):
                materials.append({
                    "id": ids[i] if i < len(ids) else None,
                    "document": documents[i] if i < len(documents) else "",
                    "metadata": metadatas[i] if i < len(metadatas) else {},
                    "source": "chroma_fallback"
                })
            
            logger.info(f"获取任务 {task_id} 的所有素材，共 {len(materials)} 条")
            return materials
            
        except Exception as e:
            logger.error(f"获取任务素材失败: {str(e)}", exc_info=True)
            return []

    def search_by_keywords_rrf(
        self,
        keywords: List[str],
        n_results_per_keyword: int = 5,
        top_k: int = 10,
        task_id: Optional[str] = None,
        distance_threshold: Optional[float] = 0.5
    ) -> Dict[str, Any]:
        """
        根据关键词列表查询，对每个关键词单独进行 RRF 后再合并
        
        流程：
        1. 对每个关键词单独进行混合检索
        2. 对每个关键词的结果进行 RRF 重排，取 top_n
        3. 合并所有关键词的结果
        4. 整体去重并排序
        
        Args:
            keywords: 关键词列表
            n_results_per_keyword: 每个关键词返回的结果数量
            top_k: 返回的最终结果数量
            task_id: 任务 ID 过滤（可选）
            distance_threshold: Chroma 距离阈值（已废弃，保留兼容性）
            
        Returns:
            包含查询关键词和素材的字典
        """
        if not keywords:
            return {
                "query_keywords": [],
                "materials": []
            }
        
        # 收集所有关键词经过 RRF 后的结果
        all_fused_results = []
        
        for keyword in keywords:
            try:
                # 步骤1：对当前关键词进行混合检索
                hybrid_result = self.hybrid_search(
                    query=keyword,
                    es_size=n_results_per_keyword,
                    chroma_size=n_results_per_keyword,
                    task_id=task_id,
                    deduplicate=False
                )
                
                # 步骤2：对当前关键词的结果进行 RRF 融合（不做预去重）
                fused_for_keyword = self.rrf_fusion(
                    es_results=hybrid_result["es_results"],
                    chroma_results=hybrid_result["chroma_results"]
                )
                
                # 取该关键词的 top_k 结果
                top_for_keyword = fused_for_keyword[:top_k]
                all_fused_results.extend(top_for_keyword)
                
                logger.debug(f"关键词 '{keyword}' 处理完成，得到 {len(top_for_keyword)} 条结果")
                
            except Exception as e:
                logger.error(f"关键词 '{keyword}' 查询失败: {str(e)}")
                continue
        
        # 返回 top_k 个结果
        materials = all_fused_results
        
        logger.info(f"关键词搜索完成，共 {len(keywords)} 个关键词，最终返回 {len(materials)} 条结果")
        
        return {
            "materials": materials
        }

    def search_with_fallback(
        self,
        keywords: List[str],
        task_id: str,
        n_results_per_keyword: int = 5,
        top_k: int = 10
    ) -> Dict[str, Any]:
        """
        搜索素材，带降级处理
        
        首先尝试使用 RRF 融合搜索，如果没有结果，则降级为获取该 task_id 的所有素材
        
        Args:
            keywords: 关键词列表
            task_id: 任务 ID（必传）
            n_results_per_keyword: 每个关键词返回的结果数量
            top_k: 返回的最终结果数量
            
        Returns:
            包含查询关键词和素材的字典
        """
        if not task_id:
            logger.warning("task_id 为空，无法进行搜索")
            return {
                "query_keywords": keywords,
                "materials": []
            }
        
        # 第一步：尝试使用 RRF 融合搜索
        logger.info(f"开始搜索，关键词: {keywords}, task_id: {task_id}")
        result = self.search_by_keywords_rrf(
            keywords=keywords,
            n_results_per_keyword=n_results_per_keyword,
            top_k=top_k,
            task_id=task_id
        )
        
        materials = result.get("materials", [])
        
        # 如果有结果，直接返回
        if materials:
            logger.info(f"搜索成功，找到 {len(materials)} 条素材")
            return result
        
        # 第二步：降级处理，获取该 task_id 的所有素材
        logger.warning(f"未找到匹配素材，降级为获取 task_id={task_id} 的所有素材")
        fallback_materials = self.get_all_task_materials(
            task_id=task_id,
            limit=top_k
        )
        
        return {
            "query_keywords": keywords,
            "materials": fallback_materials,
            "fallback": True
        }

    def search_multiple(
        self,
        queries: List[str],
        n_results_per_query: int = 3,
        task_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        多关键词混合查询，合并去重结果

        Args:
            queries: 查询文本列表
            n_results_per_query: 每个查询返回的结果数量
            task_id: 任务 ID 过滤（可选）

        Returns:
            合并后的查询结果列表，已去重
        """
        all_results = []
        seen_documents = set()

        for query in queries:
            try:
                hybrid_result = self.hybrid_search(
                    query=query,
                    es_size=n_results_per_query,
                    chroma_size=n_results_per_query,
                    task_id=task_id,
                    deduplicate=False
                )

                for res in hybrid_result["es_results"] + hybrid_result["chroma_results"]:
                    doc = res.get("document", "")
                    if doc not in seen_documents:
                        seen_documents.add(doc)
                        all_results.append(res)
            except Exception as e:
                logger.error(f"关键词 '{query}' 查询失败: {str(e)}")
                continue

        return all_results

    def search_by_keywords(
        self,
        keywords: List[str],
        n_results: int = 5,
        distance_threshold: Optional[float] = 0.5
    ) -> Dict[str, Any]:
        """
        根据关键词列表查询，返回格式化的素材字典

        Args:
            keywords: 关键词列表
            n_results: 返回结果数量
            distance_threshold: Chroma 距离阈值

        Returns:
            包含查询关键词和素材的字典
        """
        if not keywords:
            return {
                "query_keywords": [],
                "materials": []
            }

        materials = self.search_multiple(
            queries=keywords,
            n_results_per_query=n_results
        )

        if distance_threshold is not None:
            original_count = len(materials)
            filtered = []
            for mat in materials:
                if mat.get("source") == "chroma":
                    if mat.get("distance") is not None and mat["distance"] <= distance_threshold:
                        filtered.append(mat)
                else:
                    filtered.append(mat)
            materials = filtered
            logger.info(f"距离过滤: 保留 {len(filtered)}/{original_count} 条结果，阈值: {distance_threshold}")

        return {
            "query_keywords": keywords,
            "materials": materials
        }

    def count_documents(self) -> Dict[str, int]:
        """
        统计 Chroma 和 ES 中的文档数量

        Returns:
            包含 chroma_count 和 es_count 的字典
        """
        chroma_count = self.chroma_client.count()
        es_count = self.es_client.count_documents(self.index_name)
        return {
            "chroma_count": chroma_count,
            "es_count": es_count
        }

    def clear_data(self, task_id: Optional[str] = None):
        """
        清空数据

        Args:
            task_id: 如果提供则只删除该任务的数据，否则删除所有
        """
        if task_id:
            where = {"task_id": task_id}
            self.chroma_client.delete(where=where)

            must = [{"term": {"task_id": task_id}}]
            self.es_client.delete_by_query(index_name=self.index_name, body={"query": {"bool": {"must": must}}})
            logger.info(f"已清除任务 {task_id} 的数据")
        else:
            self.chroma_client.del_collection()
            self.es_client.delete_index(self.index_name)
            self._ensure_es_index_exists()
            logger.info("已清除所有数据")


def create_rag_tool(
    collection_name: Optional[str] = None,
    index_name: Optional[str] = None
) -> RAGTool:
    """
    创建 RAG 工具实例

    Args:
        collection_name: Chroma 集合名称
        index_name: ES 索引名称

    Returns:
        RAGTool 实例
    """
    return RAGTool(collection_name=collection_name, index_name=index_name)


if __name__ == "__main__":
    pass
