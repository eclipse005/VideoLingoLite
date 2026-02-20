/**
 * 文件上传组件
 * 无状态设计：上传后调用 fetchState() 刷新状态
 */

import { isValidFile, getFileType, showToast, showProgressToast, formatFileSize } from '../utils.js';

const fileInput = document.getElementById('fileInput');
const uploadArea = document.getElementById('uploadArea');

// 初始化文件上传
export function initFileUpload() {
  // 文件输入变化
  fileInput.addEventListener('change', handleFileSelect);

  // 拖拽上传
  if (uploadArea) {
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('dragleave', handleDragLeave);
    uploadArea.addEventListener('drop', handleDrop);
  }

  // 初始化 Tab 切换
  initTabSwitch();

  // 页面加载时恢复状态（轮询会自动获取文件列表）
}

/**
 * 初始化 Tab 切换功能
 */
function initTabSwitch() {
  const tabButtons = document.querySelectorAll('.tab-button');
  const tabContents = document.querySelectorAll('.tab-content');

  tabButtons.forEach(button => {
    button.addEventListener('click', () => {
      const targetTab = button.dataset.tab;

      // 移除所有 active 状态
      tabButtons.forEach(btn => btn.classList.remove('active'));
      tabContents.forEach(content => content.classList.remove('active'));

      // 添加 active 状态到当前 Tab
      button.classList.add('active');
      document.getElementById(`${targetTab}Tab`).classList.add('active');
    });
  });
}

// 处理文件选择
function handleFileSelect(event) {
  const files = Array.from(event.target.files);
  processFiles(files);
  event.target.value = ''; // 重置文件输入
}

// 处理拖拽悬停
function handleDragOver(event) {
  event.preventDefault();
  uploadArea.classList.add('drag-over');
}

// 处理拖拽离开
function handleDragLeave(event) {
  event.preventDefault();
  uploadArea.classList.remove('drag-over');
}

// 处理文件放置
function handleDrop(event) {
  event.preventDefault();
  uploadArea.classList.remove('drag-over');

  const files = Array.from(event.dataTransfer.files);
  processFiles(files);
}

// 处理文件列表
async function processFiles(files) {
  const validFiles = files.filter(file => isValidFile(file));

  if (validFiles.length === 0) {
    showToast('不支持的文件类型', 'error');
    return;
  }

  // 计算总文件大小
  const totalSize = validFiles.reduce((sum, file) => sum + file.size, 0);

  // 准备表单数据
  const formData = new FormData();
  validFiles.forEach(file => {
    formData.append('files', file);
  });

  // 小文件（< 50MB）使用普通 Toast，大文件显示进度
  const isLargeFile = totalSize >= 50 * 1024 * 1024;

  if (isLargeFile) {
    await uploadLargeFile(formData, totalSize, validFiles.length);
  } else {
    await uploadSmallFile(formData, validFiles.length);
  }
}

// 小文件上传（无进度显示）
async function uploadSmallFile(formData, fileCount) {
  showToast(`正在上传 ${fileCount} 个媒体...`, 'info');

  try {
    const response = await fetch('/api/files/upload', {
      method: 'POST',
      body: formData
    });

    if (!response.ok) {
      throw new Error(`上传失败: ${response.statusText}`);
    }

    const result = await response.json();

    // 刷新状态
    const { fetchState } = await import('../app.js');
    await fetchState();

    showToast(`成功上传 ${result.count} 个媒体`, 'success');
  } catch (error) {
    console.error('媒体上传失败:', error);
    showToast(`上传失败: ${error.message}`, 'error');
  }
}

// 大文件上传（带进度显示）
async function uploadLargeFile(formData, totalSize, fileCount) {
  const progressToast = showProgressToast(`正在上传 ${fileCount} 个媒体 (${formatFileSize(totalSize)})...`, 'info');
  let lastUpdateTime = 0;

  try {
    const result = await new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();

      // 监听上传进度（节流：每 100ms 更新一次）
      xhr.upload.addEventListener('progress', (e) => {
        if (e.lengthComputable) {
          const now = Date.now();
          if (now - lastUpdateTime > 100) {
            lastUpdateTime = now;
            const percent = Math.round((e.loaded / e.total) * 100);
            const uploaded = formatFileSize(e.loaded);
            const total = formatFileSize(e.total);
            progressToast.update(`正在上传 ${percent}% (${uploaded} / ${total})`, null, percent);
          }
        }
      });

      xhr.addEventListener('load', () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          try {
            resolve(JSON.parse(xhr.responseText));
          } catch (e) {
            reject(new Error('响应解析失败'));
          }
        } else {
          reject(new Error(`上传失败: ${xhr.statusText}`));
        }
      });

      xhr.addEventListener('error', () => reject(new Error('网络错误，上传失败')));
      xhr.addEventListener('abort', () => reject(new Error('上传已取消')));

      xhr.open('POST', '/api/files/upload');
      xhr.send(formData);
    });

    // 刷新状态
    const { fetchState } = await import('../app.js');
    await fetchState();

    // 同一个 Toast 变成成功状态
    progressToast.success(`成功上传 ${result.count} 个媒体`);
  } catch (error) {
    console.error('媒体上传失败:', error);
    // 同一个 Toast 变成错误状态
    progressToast.error(`上传失败: ${error.message}`);
  }
}
