/**
 * 文件列表组件
 * 重构后：使用 getFileTask() 获取任务状态
 */

import { getFiles, removeFile, clearAllFiles, getFileTask } from '../state.js';
import { formatFileSize, getFileIcon, escapeHtml, showToast } from '../utils.js';
import { showConfirmDialog } from './confirmDialog.js';

const fileListElement = document.getElementById('fileList');
const fileListSection = document.getElementById('fileListSection');
const fileCountElement = document.getElementById('fileCount');
const startAllBtn = document.getElementById('startAllBtn');
const clearAllBtn = document.getElementById('clearAllBtn');

// SVG 图标常量
const TRANSCRIBE_ICON = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" y1="19" x2="12" y2="23"/><line x1="8" y1="23" x2="16" y2="23"/></svg>`;
const TRANSLATE_ICON = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M5 8l6 6"/><path d="M4 14l6-6 2-3"/><path d="M2 5h12"/><path d="M7 2h1"/><path d="M22 22l-5-10-5 10"/><path d="M14 18h6"/></svg>`;
const EXPORT_ICON = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>`;
const DELETE_ICON = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></svg>`;

// 任务状态映射
const STATUS_LABELS = {
  'pending': '等待中',
  'queued': '排队中',
  'asr': 'ASR转录',
  'nlp_split': 'NLP分句',
  'hotword_correction': '热词矫正',
  'meaning_split': '语义分句',
  'summarizing': '摘要中',
  'translating': '翻译中',
  'generating': '生成字幕',
  'completed': '已完成',
  'failed': '失败',
  'cancelled': '已取消'
};

const STATUS_CLASSES = {
  'pending': 'status-pending',
  'queued': 'status-queued',
  'asr': 'status-processing',
  'nlp_split': 'status-processing',
  'hotword_correction': 'status-processing',
  'meaning_split': 'status-processing',
  'summarizing': 'status-processing',
  'translating': 'status-processing',
  'generating': 'status-processing',
  'completed': 'status-completed',
  'failed': 'status-failed',
  'cancelled': 'status-cancelled'
};

// 初始化文件列表
export function initFileList() {
  // 事件委托处理按钮点击
  fileListElement.addEventListener('click', handleFileAction);

  // 全部开始按钮
  startAllBtn.addEventListener('click', startAllFiles);

  // 清空按钮
  clearAllBtn.addEventListener('click', confirmClearAll);

  renderFileList();
}

// 处理文件操作按钮点击
function handleFileAction(event) {
  const button = event.target.closest('.file-action-btn');
  if (!button) return;

  const action = button.dataset.action;
  const fileId = button.dataset.id;

  switch (action) {
    case 'transcribe-and-translate':
      transcribeAndTranslateFile(fileId);
      break;
    case 'transcribe':
      transcribeFile(fileId);
      break;
    case 'export':
      exportFile(fileId, event);
      break;
    case 'delete':
      deleteFile(fileId);
      break;
  }
}

// 渲染文件列表
export function renderFileList() {
  const files = getFiles();

  if (files.length === 0) {
    fileListSection.style.display = 'none';
    return;
  }

  fileListSection.style.display = 'block';
  fileCountElement.textContent = `共 ${files.length} 个媒体`;

  fileListElement.innerHTML = files.map(file => {
    const task = getFileTask(file.id);  // 通过 file.id 获取任务
    const taskHtml = renderTaskInfo(task);

    return `
      <div class="file-item" data-id="${escapeHtml(file.id)}">
        <div class="file-info">
          <div class="file-icon">
            ${getFileIcon(file.type)}
          </div>
          <div class="file-details">
            <div class="file-name" title="${escapeHtml(file.name)}">${escapeHtml(file.name)}</div>
            <div class="file-meta">${formatFileSize(file.size)}</div>
            ${taskHtml}
          </div>
        </div>
        <div class="file-actions">
          <button class="file-action-btn" data-action="transcribe-and-translate" data-id="${escapeHtml(file.id)}" title="转译" ${shouldDisableButton(task) ? 'disabled' : ''}>
            ${TRANSLATE_ICON}
          </button>
          <button class="file-action-btn" data-action="transcribe" data-id="${escapeHtml(file.id)}" title="转录" ${shouldDisableButton(task) ? 'disabled' : ''}>
            ${TRANSCRIBE_ICON}
          </button>
          <button class="file-action-btn" data-action="export" data-id="${escapeHtml(file.id)}" title="导出" ${task && task.status !== 'completed' ? 'disabled' : ''}>
            ${EXPORT_ICON}
          </button>
          <button class="file-action-btn delete" data-action="delete" data-id="${escapeHtml(file.id)}" title="删除">
            ${DELETE_ICON}
          </button>
        </div>
      </div>
    `;
  }).join('');
}

// 判断按钮是否应该禁用
function shouldDisableButton(task) {
  if (!task) return false;  // 没有任务，按钮可用
  // 只有 pending/completed/failed/cancelled 状态允许点击
  return !['pending', 'completed', 'failed', 'cancelled'].includes(task.status);
}

// 渲染任务信息
function renderTaskInfo(task) {
  let html = `<div class="file-task-info">`;

  if (!task) {
    // 没有任务时显示"待开始"状态
    html += `<span class="task-status status-pending">待开始</span>`;
    html += `<span class="task-message">点击转录或转译按钮开始处理</span>`;
  } else {
    // 有任务时显示任务状态
    const statusClass = STATUS_CLASSES[task.status] || 'status-pending';
    const statusLabel = STATUS_LABELS[task.status] || task.status;

    // 状态标签
    html += `<span class="task-status ${statusClass}">${statusLabel}</span>`;

    // 进度条
    if (task.progress > 0 && task.progress < 100) {
      html += `
        <div class="task-progress-bar">
          <div class="task-progress-fill" style="width: ${task.progress}%"></div>
        </div>
        <span class="task-progress-text">${task.progress}%</span>
      `;
    }

    // 当前步骤
    if (task.currentStep) {
      html += `<span class="task-step">${escapeHtml(task.currentStep)}</span>`;
    }

    // 消息
    if (task.message && task.status !== 'completed') {
      html += `<span class="task-message">${escapeHtml(task.message)}</span>`;
    }
  }

  html += `</div>`;

  return html;
}

// 更新徽章
export function updateBadges() {
  const termsBadge = document.getElementById('termsBadge');
  const settingsBadge = document.getElementById('settingsBadge');
  const settingsBtn = document.getElementById('settingsBtn');

  if (termsBadge) {
    const termsCount = 0; // TODO: 从状态获取术语数量
    if (termsCount > 0) {
      termsBadge.textContent = termsCount;
      termsBadge.classList.add('show');
    } else {
      termsBadge.classList.remove('show');
    }
  }

  if (settingsBadge && settingsBtn) {
    const isConfigured = false; // TODO: 从状态获取配置状态
    if (isConfigured) {
      settingsBadge.classList.remove('show');
      settingsBtn.classList.remove('settings-required');
      settingsBtn.classList.add('configured');
    } else {
      settingsBadge.classList.add('show');
    }
  }
}

// 文件操作函数
async function transcribeAndTranslateFile(fileId) {
  const files = getFiles();
  const file = files.find(f => f.id === fileId);
  if (!file) return;

  try {
    console.log(`[任务] 开始转译任务: ${file.name}`);

    // 创建任务（完整流程）
    const createResponse = await fetch('/api/tasks', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        file_ids: [fileId],
        task_type: 'transcribe_and_translate'
      })
    });

    if (!createResponse.ok) {
      throw new Error(`创建任务失败: ${createResponse.statusText}`);
    }

    const createResult = await createResponse.json();
    const task = createResult.tasks[0];

    if (task) {
      console.log(`[任务] 创建任务成功:`, task);

      // 启动任务
      console.log(`[任务] 启动任务...`);
      const startResponse = await fetch(`/api/tasks/${task.id}/start`, {
        method: 'POST'
      });

      if (!startResponse.ok) {
        throw new Error(`启动任务失败: ${startResponse.statusText}`);
      }

      console.log(`[任务] 任务已启动`);
      // 刷新状态（获取最新任务状态）
      const { fetchState } = await import('../app.js');
      await fetchState();
      showToast(`任务已启动: ${file.name}`, 'success');
    }
  } catch (error) {
    console.error('处理转录和翻译任务时出错:', error);
    showToast(`处理失败: ${error.message}`, 'error');
  }
}

async function transcribeFile(fileId) {
  const files = getFiles();
  const file = files.find(f => f.id === fileId);
  if (!file) return;

  try {
    console.log(`[任务] 开始转录任务: ${file.name}`);

    // 创建任务（仅转录）
    const createResponse = await fetch('/api/tasks', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        file_ids: [fileId],
        task_type: 'transcribe_only'
      })
    });

    if (!createResponse.ok) {
      throw new Error(`创建任务失败: ${createResponse.statusText}`);
    }

    const createResult = await createResponse.json();
    const task = createResult.tasks[0];

    if (task) {
      console.log(`[任务] 创建任务成功:`, task);

      // 启动任务
      console.log(`[任务] 启动任务...`);
      const startResponse = await fetch(`/api/tasks/${task.id}/start`, {
        method: 'POST'
      });

      if (!startResponse.ok) {
        throw new Error(`启动任务失败: ${startResponse.statusText}`);
      }

      console.log(`[任务] 任务已启动`);
      // 刷新状态（获取最新任务状态）
      const { fetchState } = await import('../app.js');
      await fetchState();
      showToast(`任务已启动: ${file.name}`, 'success');
    }
  } catch (error) {
    console.error('处理转录任务时出错:', error);
    showToast(`处理失败: ${error.message}`, 'error');
  }
}

function exportFile(fileId, event) {
  const task = getFileTask(fileId);
  if (!task) {
    showToast('任务不存在', 'error');
    return;
  }

  if (task.status !== 'completed') {
    showToast('任务未完成，无法导出', 'error');
    return;
  }

  // 获取导出按钮的位置
  const button = event.currentTarget;
  const buttonRect = button.getBoundingClientRect();

  // 显示字幕选择菜单
  showSubtitleMenu(buttonRect, task.id);
}

// 显示字幕选择菜单
function showSubtitleMenu(buttonRect, taskId) {
  // 移除已存在的菜单
  const existingMenu = document.querySelector('.subtitle-export-menu');
  if (existingMenu) {
    existingMenu.remove();
  }

  // 创建菜单
  const menu = document.createElement('div');
  menu.className = 'subtitle-export-menu';

  // 字幕选项
  const subtitleOptions = [
    { type: 'src', label: '原文' },
    { type: 'trans', label: '译文' },
    { type: 'src_trans', label: '双语（上原下译）' },
    { type: 'trans_src', label: '双语（上译下原）' }
  ];

  menu.innerHTML = subtitleOptions.map(opt => `
    <div class="subtitle-export-item" data-type="${opt.type}">
      <span>${opt.label}</span>
    </div>
  `).join('');

  // 设置菜单位置（按钮右上角）
  menu.style.position = 'fixed';
  menu.style.right = `${window.innerWidth - buttonRect.right}px`;
  menu.style.bottom = `${window.innerHeight - buttonRect.top + 4}px`;

  // 添加到页面
  document.body.appendChild(menu);

  // 点击事件
  menu.addEventListener('click', (e) => {
    const item = e.target.closest('.subtitle-export-item');
    if (item) {
      const fileType = item.dataset.type;
      downloadSubtitle(taskId, fileType);
      menu.remove();
    }
  });

  // 点击其他地方关闭菜单
  const closeMenu = (e) => {
    if (!menu.contains(e.target)) {
      menu.remove();
      document.removeEventListener('click', closeMenu);
    }
  };
  setTimeout(() => {
    document.addEventListener('click', closeMenu);
  }, 0);
}

// 下载字幕
function downloadSubtitle(taskId, fileType) {
  const downloadUrl = `/api/download/${taskId}/${fileType}`;
  const link = document.createElement('a');
  link.href = downloadUrl;
  link.download = `${fileType}.srt`;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  showToast('字幕下载已开始', 'success');
}

function deleteFile(fileId) {
  const files = getFiles();
  const file = files.find(f => f.id === fileId);
  if (file) {
    showConfirmDialog(
      '删除媒体',
      `确定要删除媒体 "${file.name}" 吗？`,
      '删除',
      '取消',
      async () => {
        // 调用后端删除 API
        const response = await fetch(`/api/files/${fileId}`, {
          method: 'DELETE'
        });

        if (!response.ok) {
          showToast(`删除失败: ${file.name}`, 'error');
          return;
        }

        removeFile(fileId);
        renderFileList();
        updateBadges();
        showToast('媒体已删除', 'success');
      }
    );
  }
}

// 全部开始
async function startAllFiles() {
  const files = getFiles();
  if (files.length === 0) {
    showToast('没有可开始的媒体', 'error');
    return;
  }

  try {
    // 批量创建任务
    const createResponse = await fetch('/api/tasks', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        file_ids: files.map(f => f.id),
        task_type: 'transcribe_and_translate'
      })
    });

    if (!createResponse.ok) {
      throw new Error('创建任务失败');
    }

    const createResult = await createResponse.json();
    const tasks = createResult.tasks;

    // 按顺序启动所有任务（第一个会立即执行，其余会排队）
    for (const task of tasks) {
      const startResponse = await fetch(`/api/tasks/${task.id}/start`, {
        method: 'POST'
      });

      if (!startResponse.ok) {
        console.error(`启动任务 ${task.id} 失败`);
      }
    }

    // 刷新状态
    const { fetchState } = await import('../app.js');
    await fetchState();

    showToast(`已开始处理 ${files.length} 个媒体`, 'success');
  } catch (error) {
    console.error('全部开始失败:', error);
    showToast('全部开始失败', 'error');
  }
}

// 确认清空所有文件
function confirmClearAll() {
  const files = getFiles();
  if (files.length === 0) {
    showToast('没有可清空的媒体', 'error');
    return;
  }

  showConfirmDialog(
    '确认清空',
    `确定要清空所有 ${files.length} 个媒体吗？此操作不可恢复。`,
    '确认清空',
    '取消',
    async () => {
      // 调用后端清空 API
      const response = await fetch('/api/files', {
        method: 'DELETE'
      });

      if (!response.ok) {
        showToast('清空失败', 'error');
        return;
      }

      clearAllFiles();
      renderFileList();
      updateBadges();
      showToast('所有媒体已清空', 'success');
    }
  );
}
