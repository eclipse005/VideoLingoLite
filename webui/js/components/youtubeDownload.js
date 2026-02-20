/**
 * YouTube 下载组件
 * 处理 YouTube 视频下载和进度显示
 */

import { showToast } from '../utils.js';

export class YoutubeDownloader {
  constructor() {
    this.fileId = null;
    this.pollTimer = null;
    this.isDownloading = false;
    this.isParsing = false;
    this.parseTimer = null;
    this.availableFormats = [];

    // DOM 元素
    this.urlInput = document.getElementById('youtubeUrl');
    this.qualitySelect = document.getElementById('qualitySelect');
    this.downloadBtn = document.getElementById('downloadYoutubeBtn');
    this.progressContainer = document.getElementById('downloadProgress');
    this.progressBarFill = document.getElementById('progressBarFill');
    this.progressMessage = document.getElementById('progressMessage');
    this.progressPercent = document.getElementById('progressPercent');
    this.progressSpeed = document.getElementById('progressSpeed');
    this.progressEta = document.getElementById('progressEta');
  }

  /**
   * 初始化事件监听
   */
  init() {
    if (!this.downloadBtn) return;

    this.downloadBtn.addEventListener('click', () => this.handleDownload());

    // 支持回车键触发下载
    if (this.urlInput) {
      this.urlInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !this.isDownloading) {
          e.preventDefault();
          this.handleDownload();
        }
      });

      // 输入框失去焦点或输入完成后自动解析
      this.urlInput.addEventListener('blur', () => {
        if (this.isValidYoutubeUrl(this.urlInput.value.trim())) {
          this.parseFormats();
        }
      });

      // 防抖解析（输入停止 500ms 后触发）
      let debounceTimer;
      this.urlInput.addEventListener('input', () => {
        clearTimeout(debounceTimer);
        debounceTimer = setTimeout(() => {
          if (this.isValidYoutubeUrl(this.urlInput.value.trim())) {
            this.parseFormats();
          }
        }, 500);
      });
    }
  }

  /**
   * 处理下载按钮点击
   */
  async handleDownload() {
    if (this.isDownloading) {
      showToast('正在下载中，请稍候...', 'warning');
      return;
    }

    const url = this.urlInput?.value?.trim();

    if (!url) {
      showToast('请输入 YouTube 视频链接', 'error');
      return;
    }

    if (!this.isValidYoutubeUrl(url)) {
      showToast('请输入有效的 YouTube 链接', 'error');
      return;
    }

    // 检查是否已解析并选择质量
    if (!this.qualitySelect.value || this.qualitySelect.disabled) {
      showToast('请等待链接解析完成', 'warning');
      return;
    }

    const formatSelector = this.getFormatSelector(this.qualitySelect.value);

    await this.download(url, formatSelector);
  }

  /**
   * 验证 YouTube URL
   */
  isValidYoutubeUrl(url) {
    const pattern = /^(https?:\/\/)?(www\.)?(youtube\.com|youtu\.be)\/.+$/;
    return pattern.test(url);
  }

  /**
   * 解析视频格式
   */
  async parseFormats() {
    const url = this.urlInput?.value?.trim();

    if (!url || !this.isValidYoutubeUrl(url)) {
      return;
    }

    // 避免重复解析
    if (this.isParsing) {
      return;
    }

    this.isParsing = true;
    this.qualitySelect.disabled = true;
    this.downloadBtn.disabled = true;

    try {
      const response = await fetch('/api/files/youtube/parse', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url })
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || '解析失败');
      }

      const result = await response.json();
      this.availableFormats = result.formats || [];

      // 填充质量选择下拉框
      this.populateQualitySelect(this.availableFormats);

      // 启用下载按钮
      this.downloadBtn.disabled = false;

    } catch (error) {
      console.error('解析视频格式失败:', error);
      showToast(error.message, 'error');

      // 解析失败时，重置为"画质"状态
      this.qualitySelect.innerHTML = '<option value="" disabled selected>画质</option>';
      this.qualitySelect.disabled = true;
      this.downloadBtn.disabled = true;
    } finally {
      this.isParsing = false;
    }
  }

  /**
   * 填充质量选择下拉框
   */
  populateQualitySelect(formats) {
    if (!this.qualitySelect || formats.length === 0) {
      // 无可用格式时，重置为"画质"状态
      this.qualitySelect.innerHTML = '<option value="" disabled selected>画质</option>';
      this.qualitySelect.disabled = true;
      this.downloadBtn.disabled = true;
      return;
    }

    // 清空并重建选项
    this.qualitySelect.innerHTML = '';

    // 添加格式选项（第一个是"最佳"）
    formats.forEach(format => {
      const option = document.createElement('option');
      option.value = format.format_selector;
      option.textContent = format.label;

      // 标记推荐选项（默认选中）
      if (format.is_recommended) {
        option.selected = true;
      }

      this.qualitySelect.appendChild(option);
    });

    this.qualitySelect.disabled = false;
  }

  /**
   * 根据选中的格式获取 format_selector
   */
  getFormatSelector(selectedValue) {
    // 直接返回选中的值，因为"最佳"也有对应的 format_selector
    return selectedValue;
  }

  /**
   * 开始下载
   */
  async download(url, formatSelector) {
    this.isDownloading = true;
    this.setDownloadButtonState(true);
    this.showProgress(true);
    this.updateProgress({ message: '开始下载...', progress: 0 });

    try {
      const response = await fetch('/api/files/youtube', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          url,
          format_selector: formatSelector
        })
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || '下载请求失败');
      }

      const result = await response.json();
      this.fileId = result.file_id;

      // 开始轮询进度
      this.startPolling();

    } catch (error) {
      showToast(error.message, 'error');
      this.showProgress(false);
      this.isDownloading = false;
      this.setDownloadButtonState(false);
    }
  }

  /**
   * 开始轮询进度
   */
  startPolling() {
    if (this.pollTimer) {
      clearInterval(this.pollTimer);
    }

    // 立即查询一次
    this.fetchProgress();

    // 每秒轮询
    this.pollTimer = setInterval(() => {
      this.fetchProgress();
    }, 1000);
  }

  /**
   * 获取下载进度
   */
  async fetchProgress() {
    if (!this.fileId) return;

    try {
      const response = await fetch(`/api/files/youtube/progress/${this.fileId}`);

      if (!response.ok) {
        throw new Error('获取进度失败');
      }

      const progress = await response.json();
      this.updateProgress(progress);

      // 检查下载状态
      if (progress.status === 'completed') {
        this.stopPolling();
        this.handleDownloadComplete();
      } else if (progress.status === 'error') {
        this.stopPolling();
        this.handleDownloadError(progress.message || '下载失败');
      }

    } catch (error) {
      console.error('获取进度失败:', error);
    }
  }

  /**
   * 停止轮询
   */
  stopPolling() {
    if (this.pollTimer) {
      clearInterval(this.pollTimer);
      this.pollTimer = null;
    }
  }

  /**
   * 更新进度显示
   */
  updateProgress(progress) {
    if (!progress) return;

    // 更新进度条
    const percent = progress.progress || 0;
    this.progressBarFill.style.width = `${percent}%`;
    this.progressPercent.textContent = `${Math.round(percent)}%`;
    this.progressMessage.textContent = progress.message || '下载中...';

    // 更新速度
    if (progress.speed && progress.speed > 0) {
      const speedMB = (progress.speed / 1024 / 1024).toFixed(1);
      this.progressSpeed.textContent = `${speedMB} MB/s`;
    } else {
      this.progressSpeed.textContent = '-- MB/s';
    }

    // 更新剩余时间
    if (progress.eta && progress.eta > 0) {
      const minutes = Math.floor(progress.eta / 60);
      const seconds = Math.round(progress.eta % 60);
      this.progressEta.textContent = `剩余 ${minutes}:${seconds.toString().padStart(2, '0')}`;
    } else {
      this.progressEta.textContent = '剩余 --:--';
    }
  }

  /**
   * 下载完成处理
   */
  handleDownloadComplete() {
    showToast('视频下载完成！', 'success');

    // 延迟隐藏进度条
    setTimeout(() => {
      this.showProgress(false);
      this.isDownloading = false;
      this.setDownloadButtonState(false);

      // 清空输入框和重置下拉框
      if (this.urlInput) {
        this.urlInput.value = '';
      }
      if (this.qualitySelect) {
        this.qualitySelect.innerHTML = '<option value="" disabled selected>画质</option>';
        this.qualitySelect.disabled = true;
      }
      this.downloadBtn.disabled = true;

      // 触发文件列表刷新（会自动显示新下载的文件）
      window.dispatchEvent(new CustomEvent('files-changed'));
    }, 2000);
  }

  /**
   * 下载错误处理
   */
  handleDownloadError(message) {
    showToast(message, 'error');
    this.showProgress(false);
    this.isDownloading = false;
    this.setDownloadButtonState(false);
  }

  /**
   * 显示/隐藏进度条
   */
  showProgress(show) {
    if (this.progressContainer) {
      this.progressContainer.style.display = show ? 'block' : 'none';
    }
  }

  /**
   * 设置下载按钮状态
   */
  setDownloadButtonState(downloading) {
    if (!this.downloadBtn) return;

    if (downloading) {
      this.downloadBtn.disabled = true;
      this.downloadBtn.innerHTML = `
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" class="spinning">
          <path d="M21 12a9 9 0 1 1-6.219-8.56"/>
        </svg>
        下载中...
      `;
    } else {
      this.downloadBtn.disabled = false;
      this.downloadBtn.innerHTML = `
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
          <polyline points="7 10 12 15 17 10"/>
          <line x1="12" y1="15" x2="12" y2="3"/>
        </svg>
        下载视频
      `;
    }
  }
}

// 创建全局实例
export const youtubeDownloader = new YoutubeDownloader();
