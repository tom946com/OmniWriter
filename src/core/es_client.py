import os
from typing import Any, Dict, List, Optional, Union
from elasticsearch import Elasticsearch
from dotenv import load_dotenv
from src.utils.logs import logger

load_dotenv()


class ESClient:
    """
    Elasticsearch客户端封装类，提供索引管理及文档增删改查功能
    """

    def __init__(
        self,
        hosts: Optional[Union[str, List[str]]] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        scheme: Optional[str] = None,
        port: Optional[int] = None,
        verify_certs: bool = False,
    ):
        """
        初始化Elasticsearch客户端

        :param hosts: ES主机地址，支持单个或多个地址
        :param username: 用户名
        :param password: 密码
        :param scheme: 协议（http/https）
        :param port: 端口号
        :param verify_certs: 是否验证SSL证书
        """
        self.hosts = hosts or os.getenv("ES_HOSTS", "http://localhost:9200")
        self.username = username or os.getenv("ES_USERNAME", "elastic")
        self.password = password or os.getenv("ES_PASSWORD", "")
        self.scheme = scheme or os.getenv("ES_SCHEME", "http")
        self.port = port or int(os.getenv("ES_PORT", "9200"))
        self.verify_certs = verify_certs

        self.client = self._create_client()

    def _create_client(self) -> Elasticsearch:
        """
        创建Elasticsearch连接实例

        :return: Elasticsearch实例
        """
        if isinstance(self.hosts, str):
            hosts_list = [self.hosts]
        else:
            hosts_list = self.hosts

        client = Elasticsearch(
            hosts=hosts_list,
            basic_auth=(self.username, self.password) if self.password else None,
            verify_certs=self.verify_certs,
            request_timeout=180,
            timeout=180,
            max_retries=3,
            retry_on_timeout=True,
        )

        try:
            if not client.ping():
                raise ConnectionError("无法连接到Elasticsearch集群")
            logger.info(f"成功连接到Elasticsearch集群，版本: {client.info()['version']['number']}")
        except Exception as e:
            logger.error(f"Elasticsearch连接失败: {e}")
            raise

        return client

    def ping(self) -> bool:
        """
        检查ES连接状态

        :return: 连接是否正常
        """
        return self.client.ping()

    def get_info(self) -> Dict[str, Any]:
        """
        获取ES集群信息

        :return: 集群信息字典
        """
        return self.client.info()

    # ==================== Index Management ====================

    def create_index(
        self,
        index_name: str,
        mappings: Optional[Dict[str, Any]] = None,
        settings: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        创建索引

        :param index_name: 索引名称
        :param mappings: 映射配置
        :param settings: 索引设置
        :return: 是否创建成功
        """
        body: Dict[str, Any] = {}
        if settings:
            body["settings"] = settings
        if mappings:
            body["mappings"] = mappings

        try:
            if self.client.indices.exists(index=index_name):
                logger.warning(f"索引 {index_name} 已存在")
                return False

            self.client.indices.create(index=index_name, body=body)
            return True
        except Exception as e:
            logger.error(f"创建索引失败: {e}")
            raise

    def delete_index(self, index_name: str) -> bool:
        """
        删除索引

        :param index_name: 索引名称
        :return: 是否删除成功
        """
        try:
            if not self.client.indices.exists(index=index_name):
                logger.warning(f"索引 {index_name} 不存在")
                return False

            self.client.indices.delete(index=index_name)
            logger.info(f"成功删除索引: {index_name}")
            return True
        except Exception as e:
            logger.error(f"删除索引失败: {e}")
            raise

    def index_exists(self, index_name: str) -> bool:
        """
        检查索引是否存在

        :param index_name: 索引名称
        :return: 是否存在
        """
        return self.client.indices.exists(index=index_name)

    def get_index_mapping(self, index_name: str) -> Dict[str, Any]:
        """
        获取索引映射

        :param index_name: 索引名称
        :return: 映射信息
        """
        return self.client.indices.get_mapping(index=index_name)

    # ==================== Document CRUD ====================

    def create_document(
        self,
        index_name: str,
        document: Dict[str, Any],
        doc_id: Optional[str] = None,
        refresh: str = "wait_for",
    ) -> Dict[str, Any]:
        """
        创建文档（Create）

        :param index_name: 索引名称
        :param document: 文档内容
        :param doc_id: 文档ID，如果不指定则自动生成
        :param refresh: 刷新策略（true/false/wait_for）
        :return: 创建结果
        """
        try:
            response = self.client.index(
                index=index_name,
                id=doc_id,
                document=document,
                op_type="create",
                refresh=refresh,
            )
            logger.info(f"文档创建成功，ID: {response.get('_id')}, 索引: {index_name}")
            return response
        except Exception as e:
            logger.error(f"创建文档失败: {e}")
            raise

    def get_document(
        self,
        index_name: str,
        doc_id: str,
    ) -> Optional[Dict[str, Any]]:
        """
        获取文档（Read）

        :param index_name: 索引名称
        :param doc_id: 文档ID
        :return: 文档内容
        """
        try:
            response = self.client.get(index=index_name, id=doc_id)
            logger.debug(f"获取文档成功，ID: {doc_id}, 索引: {index_name}")
            return {
                "_id": response["_id"],
                "_source": response["_source"],
                "_version": response.get("_version"),
            }
        except Exception as e:
            if "not_found" in str(e).lower():
                logger.warning(f"文档未找到，ID: {doc_id}, 索引: {index_name}")
                return None
            logger.error(f"获取文档失败: {e}")
            raise

    def update_document(
        self,
        index_name: str,
        doc_id: str,
        document: Dict[str, Any],
        refresh: str = "wait_for",
    ) -> Dict[str, Any]:
        """
        更新文档（Update）

        :param index_name: 索引名称
        :param doc_id: 文档ID
        :param document: 需要更新的字段内容
        :param refresh: 刷新策略
        :return: 更新结果
        """
        try:
            body = {"doc": document}
            response = self.client.update(
                index=index_name,
                id=doc_id,
                body=body,
                refresh=refresh,
            )
            logger.info(f"文档更新成功，ID: {doc_id}, 索引: {index_name}")
            return response
        except Exception as e:
            logger.error(f"更新文档失败: {e}")
            raise

    def upsert_document(
        self,
        index_name: str,
        document: Dict[str, Any],
        doc_id: Optional[str] = None,
        refresh: str = "wait_for",
    ) -> Dict[str, Any]:
        """
        插入或更新文档（Upsert）

        :param index_name: 索引名称
        :param document: 文档内容
        :param doc_id: 文档ID
        :param refresh: 刷新策略
        :return: 操作结果
        """
        try:
            response = self.client.index(
                index=index_name,
                id=doc_id,
                document=document,
                refresh=refresh,
                timeout="180s",
            )
            result = "updated" if response.get("result") == "updated" else "created"
            logger.info(f"文档{result}成功，ID: {response.get('_id')}, 索引: {index_name}")
            return response
        except Exception as e:
            logger.error(f"插入或更新文档失败: {e}")
            raise

    def delete_document(
        self,
        index_name: str,
        doc_id: str,
        refresh: str = "wait_for",
    ) -> bool:
        """
        删除文档（Delete）

        :param index_name: 索引名称
        :param doc_id: 文档ID
        :param refresh: 刷新策略
        :return: 是否删除成功
        """
        try:
            response = self.client.delete(
                index=index_name,
                id=doc_id,
                refresh=refresh,
                timeout="180s"
            )
            logger.info(f"文档删除成功，ID: {doc_id}, 索引: {index_name}")
            return response.get("result") == "deleted"
        except Exception as e:
            if "not_found" in str(e).lower():
                logger.warning(f"文档未找到，无法删除，ID: {doc_id}")
                return False
            logger.error(f"删除文档失败: {e}")
            raise

    # ==================== Bulk Operations ====================

    def bulk_create_documents(
        self,
        index_name: str,
        documents: List[Dict[str, Any]],
        ids: Optional[List[str]] = None,
        refresh: str = "wait_for",
    ) -> Dict[str, Any]:
        """
        批量创建文档

        :param index_name: 索引名称
        :param documents: 文档列表
        :param ids: 文档ID列表
        :param refresh: 刷新策略
        :return: 批量操作结果
        """
        actions = []
        for idx, doc in enumerate(documents):
            action = {"_index": index_name, "_op_type": "create", "_source": doc}
            if ids and idx < len(ids):
                action["_id"] = ids[idx]
            actions.append(action)

        try:
            from elasticsearch.helpers import bulk
            success, errors = bulk(self.client, actions, refresh=refresh)
            logger.info(f"批量创建完成，成功: {success} 条，错误: {len(errors)} 条")
            return {"success": success, "errors": errors}
        except Exception as e:
            logger.error(f"批量创建文档失败: {e}")
            raise

    def bulk_update_documents(
        self,
        index_name: str,
        documents: List[Dict[str, Any]],
        ids: List[str],
        refresh: str = "wait_for",
    ) -> Dict[str, Any]:
        """
        批量更新文档

        :param index_name: 索引名称
        :param documents: 更新的字段列表
        :param ids: 对应的文档ID列表
        :param refresh: 刷新策略
        :return: 批量操作结果
        """
        actions = []
        for doc_id, doc in zip(ids, documents):
            action = {
                "_index": index_name,
                "_op_type": "update",
                "_id": doc_id,
                "doc": doc,
            }
            actions.append(action)

        try:
            from elasticsearch.helpers import bulk
            success, errors = bulk(self.client, actions, refresh=refresh)
            logger.info(f"批量更新完成，成功: {success} 条，错误: {len(errors)} 条")
            return {"success": success, "errors": errors}
        except Exception as e:
            logger.error(f"批量更新文档失败: {e}")
            raise

    def bulk_delete_documents(
        self,
        index_name: str,
        doc_ids: List[str],
        refresh: str = "wait_for",
    ) -> Dict[str, Any]:
        """
        批量删除文档

        :param index_name: 索引名称
        :param doc_ids: 要删除的文档ID列表
        :param refresh: 刷新策略
        :return: 批量操作结果
        """
        actions = [
            {"_index": index_name, "_op_type": "delete", "_id": doc_id}
            for doc_id in doc_ids
        ]

        try:
            from elasticsearch.helpers import bulk
            success, errors = bulk(self.client, actions, refresh=refresh)
            logger.info(f"批量删除完成，成功: {success} 条，错误: {len(errors)} 条")
            return {"success": success, "errors": errors}
        except Exception as e:
            logger.error(f"批量删除文档失败: {e}")
            raise

    # ==================== Search Operations ====================

    def search(
        self,
        index_name: str,
        query: Optional[Dict[str, Any]] = None,
        size: int = 10,
        from_: int = 0,
        sort: Optional[List[Any]] = None,
        source: Optional[List[str]] = None,
        highlight: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        通用搜索接口

        :param index_name: 索引名称
        :param query: DSL查询语句
        :param size: 返回数量
        :param from_: 分页起始位置
        :param sort: 排序规则
        :param source: 返回字段过滤
        :param highlight: 高亮配置
        :return: 搜索结果
        """
        body: Dict[str, Any] = {
            "query": query or {"match_all": {}},
            "size": size,
            "from": from_,
        }

        if sort:
            body["sort"] = sort
        if source:
            body["_source"] = source
        if highlight:
            body["highlight"] = highlight

        body.update(kwargs)

        try:
            response = self.client.search(index=index_name, body=body, timeout="180s")
            hits = response.get("hits", {})
            logger.debug(
                f"搜索完成，命中: {hits.get('total', {}).get('value', 0)} 条记录"
            )
            return response
        except Exception as e:
            logger.error(f"搜索失败: {e}")
            raise

    def match_query(
        self,
        index_name: str,
        field: str,
        value: Any,
        size: int = 10,
        from_: int = 0,
        operator: str = "or",
    ) -> Dict[str, Any]:
        """
        Match查询（全文搜索）

        :param index_name: 索引名称
        :param field: 查询字段
        :param value: 查询值
        :param size: 返回数量
        :param from_: 分页起始位置
        :param operator: 操作符（or/and）
        :return: 搜索结果
        """
        query = {
            "match": {
                field: {
                    "query": value,
                    "operator": operator,
                }
            }
        }
        return self.search(index_name, query=query, size=size, from_=from_)

    def term_query(
        self,
        index_name: str,
        field: str,
        value: Any,
        size: int = 10,
        from_: int = 0,
    ) -> Dict[str, Any]:
        """
        Term查询（精确匹配）

        :param index_name: 索引名称
        :param field: 查询字段
        :param value: 查询值
        :param size: 返回数量
        :param from_: 分页起始位置
        :return: 搜索结果
        """
        query = {"term": {field: value}}
        return self.search(index_name, query=query, size=size, from_=from_)

    def range_query(
        self,
        index_name: str,
        field: str,
        gte: Optional[Any] = None,
        lte: Optional[Any] = None,
        gt: Optional[Any] = None,
        lt: Optional[Any] = None,
        size: int = 10,
        from_: int = 0,
    ) -> Dict[str, Any]:
        """
        Range查询（范围查询）

        :param index_name: 索引名称
        :param field: 查询字段
        :param gte: 大于等于
        :param lte: 小于等于
        :param gt: 大于
        :param lt: 小于
        :param size: 返回数量
        :param from_: 分页起始位置
        :return: 搜索结果
        """
        range_params = {}
        if gte is not None:
            range_params["gte"] = gte
        if lte is not None:
            range_params["lte"] = lte
        if gt is not None:
            range_params["gt"] = gt
        if lt is not None:
            range_params["lt"] = lt

        query = {"range": {field: range_params}}
        return self.search(index_name, query=query, size=size, from_=from_)

    def bool_query(
        self,
        index_name: str,
        must: Optional[List[Dict]] = None,
        must_not: Optional[List[Dict]] = None,
        should: Optional[List[Dict]] = None,
        filter: Optional[List[Dict]] = None,
        minimum_should_match: Optional[int] = None,
        size: int = 10,
        from_: int = 0,
    ) -> Dict[str, Any]:
        """
        Bool查询（组合查询）

        :param index_name: 索引名称
        :param must: 必须满足的条件列表
        :param must_not: 必须不满足的条件列表
        :param should: 应该满足的条件列表
        :param filter: 过滤条件列表
        :param minimum_should_match: 最少满足should条件的数量
        :param size: 返回数量
        :param from_: 分页起始位置
        :return: 搜索结果
        """
        query: Dict[str, Any] = {"bool": {}}

        if must:
            query["bool"]["must"] = must
        if must_not:
            query["bool"]["must_not"] = must_not
        if should:
            query["bool"]["should"] = should
        if filter:
            query["bool"]["filter"] = filter
        if minimum_should_match is not None:
            query["bool"]["minimum_should_match"] = minimum_should_match

        return self.search(index_name, query=query, size=size, from_=from_)

    def multi_match_query(
        self,
        index_name: str,
        query_text: str,
        fields: List[str],
        size: int = 10,
        from_: int = 0,
        operator: str = "or",
        type_: str = "best_fields",
    ) -> Dict[str, Any]:
        """
        Multi-Match查询（多字段搜索）

        :param index_name: 索引名称
        :param query_text: 查询文本
        :param fields: 查询字段列表
        :param size: 返回数量
        :param from_: 分页起始位置
        :param operator: 操作符
        :param type_: 匹配类型
        :return: 搜索结果
        """
        query = {
            "multi_match": {
                "query": query_text,
                "fields": fields,
                "operator": operator,
                "type": type_,
            }
        }
        return self.search(index_name, query=query, size=size, from_=from_)

    def scroll_search(
        self,
        index_name: str,
        query: Optional[Dict[str, Any]] = None,
        scroll_size: int = 1000,
        scroll_time: str = "5m",
    ) -> List[Dict[str, Any]]:
        """
        Scroll搜索（用于大量数据查询）

        :param index_name: 索引名称
        :param query: DSL查询语句
        :param scroll_size: 每次滚动返回的数量
        :param scroll_time: 游标保留时间
        :return: 所有匹配的文档列表
        """
        all_docs = []

        try:
            response = self.client.search(
                index=index_name,
                body={"query": query or {"match_all": {}}, "size": scroll_size},
                scroll=scroll_time,
            )
            scroll_id = response.get("_scroll_id")
            hits = response["hits"]["hits"]

            while hits:
                all_docs.extend(hits)
                response = self.client.scroll(scroll_id=scroll_id, scroll=scroll_time)
                scroll_id = response.get("_scroll_id")
                hits = response["hits"]["hits"]

            self.client.clear_scroll(scroll_id=scroll_id)
            logger.info(f"Scroll搜索完成，共获取 {len(all_docs)} 条记录")

            return all_docs
        except Exception as e:
            logger.error(f"Scroll搜索失败: {e}")
            raise

    def count_documents(
        self,
        index_name: str,
        query: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        统计文档数量

        :param index_name: 索引名称
        :param query: 查询条件
        :return: 文档数量
        """
        try:
            response = self.client.count(
                index=index_name,
                body={"query": query} if query else None,
            )
            count = response.get("count", 0)
            logger.debug(f"索引 {index_name} 中共有 {count} 条文档")
            return count
        except Exception as e:
            logger.error(f"统计文档数量失败: {e}")
            raise

    def delete_by_query(
        self,
        index_name: str,
        body: Dict[str, Any],
        refresh: str = "wait_for",
    ) -> int:
        """
        根据查询条件删除文档

        :param index_name: 索引名称
        :param body: 查询条件
        :param refresh: 刷新策略
        :return: 删除的文档数量
        """
        try:
            response = self.client.delete_by_query(
                index=index_name,
                body=body,
                refresh=refresh,
            )
            deleted = response.get("deleted", 0)
            logger.info(f"删除查询完成，删除 {deleted} 条文档")
            return deleted
        except Exception as e:
            logger.error(f"删除查询失败: {e}")
            raise

    # ==================== Alias Operations ====================

    def add_alias(self, index_name: str, alias_name: str) -> bool:
        """
        为索引添加别名

        :param index_name: 索引名称
        :param alias_name: 别名
        :return: 是否添加成功
        """
        try:
            self.client.indices.put_alias(index=index_name, name=alias_name)
            logger.info(f"为索引 {index_name} 添加别名 {alias_name} 成功")
            return True
        except Exception as e:
            logger.error(f"添加别名失败: {e}")
            raise

    def remove_alias(self, index_name: str, alias_name: str) -> bool:
        """
        移除索引别名

        :param index_name: 索引名称
        :param alias_name: 别名
        :return: 是否移除成功
        """
        try:
            self.client.indices.delete_alias(index=index_name, name=alias_name)
            logger.info(f"从索引 {index_name} 移除别名 {alias_name} 成功")
            return True
        except Exception as e:
            logger.error(f"移除别名失败: {e}")
            raise

    # ==================== Close ====================

    def close(self):
        """关闭ES连接"""
        try:
            self.client.close()
            logger.info("Elasticsearch连接已关闭")
        except Exception as e:
            logger.error(f"关闭连接时出错: {e}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
