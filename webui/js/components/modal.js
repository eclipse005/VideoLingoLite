/**
 * 模态框组件
 */

import { escapeHtml, showToast } from '../utils.js';

const modalContainer = document.getElementById('modalContainer');

// 模态框内容生成器
function getModalContent(type) {
  switch(type) {
    case 'terms':
      return `
        <h2 class="apple-heading-medium" style="margin-bottom: 24px;">术语管理</h2>
        <p class="apple-body">术语列表为空</p>
        <button class="apple-button" style="margin-top: 16px;">添加术语</button>
      `;
    case 'settings':
      return `
        <h2 class="apple-heading-medium" style="margin-bottom: 24px;">设置</h2>
        <div class="settings-form">
          <div style="margin-bottom: 20px;">
            <label class="apple-body" style="display: block; margin-bottom: 8px;">API Key</label>
            <input type="password" placeholder="输入你的 API Key" style="width: 100%; padding: 12px; border: 1px solid var(--apple-border-light); border-radius: 8px; font-size: 15px;">
          </div>
          <div style="margin-bottom: 20px;">
            <label class="apple-body" style="display: block; margin-bottom: 8px;">目标语言</label>
            <select style="width: 100%; padding: 12px; border: 1px solid var(--apple-border-light); border-radius: 8px; font-size: 15px;">
              <option>简体中文</option>
              <option>English</option>
              <option>日本語</option>
            </select>
          </div>
          <button class="apple-button" onclick="saveSettings()">保存设置</button>
        </div>
      `;
  }
}

// 打开模态框
export function openModal(type) {
  const modalContent = getModalContent(type);
  modalContainer.innerHTML = `
    <div class="modal-overlay" onclick="closeModal()">
      <div class="modal-content" onclick="event.stopPropagation()">
        <button class="modal-close" onclick="closeModal()">&times;</button>
        ${modalContent}
      </div>
    </div>
  `;
  document.body.style.overflow = 'hidden';
}

// 关闭模态框
export function closeModal() {
  modalContainer.innerHTML = '';
  document.body.style.overflow = '';
}

// 保存设置（供 HTML onclick 调用）
window.saveSettings = function() {
  // TODO: 实现保存逻辑
  closeModal();
  showToast('设置已保存', 'success');
};

// 导出到 window 供 HTML 使用
window.openModal = openModal;
window.closeModal = closeModal;
