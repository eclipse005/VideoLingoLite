#!/usr/bin/env python
"""
VideoLingoLite 统一启动脚本
同时启动后端 API 和前端服务
"""

import uvicorn
import os
import sys
import subprocess
import time
import threading
import webbrowser

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)


def open_browser():
    """延迟打开浏览器"""
    time.sleep(2)  # 等待服务器启动
    webbrowser.open("http://localhost:8000")


def main():
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║            VideoLingoLite - 音视频转写翻译工具               ║
    ║                                                               ║
    ║  🌐 Web UI:    http://localhost:8000                         ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)

    # 创建必要的目录
    print("📁 创建必要的目录...")
    os.makedirs("api/uploads", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    os.makedirs("api/logs", exist_ok=True)  # 日志目录
    print("✅ 目录创建完成")

    # 在后台线程中打开浏览器
    browser_thread = threading.Thread(target=open_browser, daemon=True)
    browser_thread.start()

    # 启动服务
    print("\n🚀 启动服务...")
    print("=" * 60)

    try:
        # 启动 FastAPI（已配置静态文件服务，前端自动可用）
        uvicorn.run(
            "api.main:app",
            host="0.0.0.0",
            port=8000,
            reload=False,  # 关闭 reload 以避免多进程内存不同步问题
            log_level="info",
            access_log=False  # 禁用 uvicorn 的访问日志
        )
    except KeyboardInterrupt:
        print("\n\n👋 服务已停止")
    except Exception as e:
        print(f"\n❌ 启动失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
