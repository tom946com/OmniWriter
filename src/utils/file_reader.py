"""
文件读取工具模块

提供通用的文件读取功能，支持多种文件格式
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class FileReader:
    """
    文件读取工具类
    
    支持读取多种格式的文件内容，返回统一的字典格式
    """
    
    SUPPORTED_EXTENSIONS = {'.txt', '.md', '.markdown', '.csv', '.json', '.xml', '.html', '.htm'}
    
    @classmethod
    def read_file(cls, file_path: str) -> Dict[str, str]:
        """
        读取单个文件的内容
        
        Args:
            file_path: 文件的绝对路径或相对路径
            
        Returns:
            包含文件信息的字典:
            {
                "filename": 文件名,
                "filepath": 文件路径,
                "content": 文件内容,
                "extension": 文件扩展名
            }
            
        Raises:
            FileNotFoundError: 文件不存在
            ValueError: 不支持的文件格式
            IOError: 文件读取错误
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        if not path.is_file():
            raise IOError(f"路径不是文件: {file_path}")
        
        extension = path.suffix.lower()
        filename = path.name
        
        try:
            if extension in cls.SUPPORTED_EXTENSIONS:
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
            else:
                raise ValueError(f"不支持的文件格式: {extension}")
            
            return {
                "filename": filename,
                "filepath": str(path.absolute()),
                "content": content,
                "extension": extension
            }
            
        except UnicodeDecodeError:
            try:
                with open(path, 'r', encoding='gbk') as f:
                    content = f.read()
                
                return {
                    "filename": filename,
                    "filepath": str(path.absolute()),
                    "content": content,
                    "extension": extension
                }
            except Exception as e:
                raise IOError(f"无法解码文件 {filename}: {str(e)}")
    
    @classmethod
    def read_files(cls, file_paths: List[str]) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
        """
        批量读取多个文件
        
        Args:
            file_paths: 文件路径列表
            
        Returns:
            元组 (成功列表, 失败列表):
            - 成功列表: 包含成功读取的文件信息字典
            - 失败列表: 包含失败信息 {"path": 路径, "error": 错误信息}
        """
        success_results = []
        failed_results = []
        
        for file_path in file_paths:
            try:
                result = cls.read_file(file_path)
                success_results.append(result)
            except Exception as e:
                failed_results.append({
                    "path": file_path,
                    "error": str(e)
                })
        
        return success_results, failed_results
    
    @classmethod
    def is_supported_format(cls, file_path: str) -> bool:
        """
        检查文件格式是否支持
        
        Args:
            file_path: 文件路径
            
        Returns:
            bool: 是否支持该文件格式
        """
        extension = Path(file_path).suffix.lower()
        return extension in cls.SUPPORTED_EXTENSIONS
    
    @classmethod
    def get_file_info(cls, file_path: str) -> Dict[str, any]:
        """
        获取文件基本信息（不读取内容）
        
        Args:
            file_path: 文件路径
            
        Returns:
            包含文件基本信息的字典
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        stat = path.stat()
        
        return {
            "filename": path.name,
            "filepath": str(path.absolute()),
            "extension": path.suffix.lower(),
            "size_bytes": stat.st_size,
            "is_supported": cls.is_supported_format(file_path)
        }


def read_single_file(file_path: str) -> Dict[str, str]:
    """
    便捷函数：读取单个文件
    
    Args:
        file_path: 文件路径
        
    Returns:
        文件信息字典
    """
    return FileReader.read_file(file_path)


def read_multiple_files(file_paths: List[str]) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """
    便捷函数：批量读取多个文件
    
    Args:
        file_paths: 文件路径列表
        
    Returns:
        (成功结果列表, 失败结果列表)
    """
    return FileReader.read_files(file_paths)


if __name__ == "__main__":
    test_files = [
        "test.txt",
        "example.md"
    ]
    
    print("测试文件读取功能:")
    print("=" * 60)
    
    for test_file in test_files:
        print(f"\n测试文件: {test_file}")
        if FileReader.is_supported_format(test_file):
            try:
                info = FileReader.get_file_info(test_file)
                print(f"文件信息: {info}")
                
                result = FileReader.read_file(test_file)
                print(f"文件名: {result['filename']}")
                print(f"内容长度: {len(result['content'])} 字符")
                print(f"前100个字符预览: {result['content'][:100]}...")
            except FileNotFoundError:
                print("文件不存在")
            except Exception as e:
                print(f"错误: {e}")
        else:
            print("不支持的文件格式")