/**
 * 确认对话框组件
 */

import { escapeHtml } from '../utils.js';
import { getConfirmDialog, showConfirmDialog as setConfirmDialog, hideConfirmDialog } from '../state.js';

const modalContainer = document.getElementById('modalContainer');

let currentOnConfirm = null;

// 渲染确认对话框
function renderConfirmDialog() {
  const dialog = getConfirmDialog();

  if (!dialog) {
    modalContainer.innerHTML = '';
    document.body.style.overflow = '';
    return;
  }

  const { title, message, confirmText, cancelText, confirmButtonClass } = dialog;

  modalContainer.innerHTML = `
    <div class="confirm-dialog-overlay" onclick="window.closeConfirmDialog()">
      <div class="confirm-dialog-content" onclick="event.stopPropagation()">
        <div class="confirm-dialog-header">
          <h3 class="apple-heading-small">${escapeHtml(title)}</h3>
          <button class="confirm-dialog-close" onclick="window.closeConfirmDialog()" aria-label="关闭">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <line x1="18" y1="6" x2="6" y2="18"/>
              <line x1="6" y1="6" x2="18" y2="18"/>
            </svg>
          </button>
        </div>
        <div class="confirm-dialog-body">
          <p class="confirm-dialog-message">${escapeHtml(message)}</p>
        </div>
        <div class="confirm-dialog-footer">
          <button class="apple-button apple-button-ghost" onclick="window.cancelConfirm()">${escapeHtml(cancelText)}</button>
          <button class="apple-button apple-button-${confirmButtonClass}" onclick="window.confirmAction()">${escapeHtml(confirmText)}</button>
        </div>
      </div>
    </div>
  `;
  document.body.style.overflow = 'hidden';
}

// 显示确认对话框
export function showConfirmDialog(title, message, confirmText = '确认', cancelText = '取消', onConfirm, confirmButtonClass = 'danger') {
  const dialog = {
    title,
    message,
    confirmText,
    cancelText,
    onConfirm,
    confirmButtonClass
  };

  // 保存回调
  currentOnConfirm = onConfirm;

  // 保存到状态并渲染
  setConfirmDialog(dialog);
  renderConfirmDialog();
}

// 确认操作
window.confirmAction = function() {
  if (currentOnConfirm) {
    currentOnConfirm();
    currentOnConfirm = null;
  }
  hideConfirmDialog();
  renderConfirmDialog();
};

// 取消操作
window.cancelConfirm = function() {
  currentOnConfirm = null;
  hideConfirmDialog();
  renderConfirmDialog();
};

// 关闭确认对话框
window.closeConfirmDialog = function() {
  currentOnConfirm = null;
  hideConfirmDialog();
  renderConfirmDialog();
};
