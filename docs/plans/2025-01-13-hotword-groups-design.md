# 热词分组功能设计文档

**日期**: 2025-01-13
**状态**: 设计完成
**参考**: `KeytermGroupsSettings.tsx`

---

## 概述

为 VideoLingoLite 的热词矫正功能添加分组管理能力，支持按领域组织热词，提升专业术语识别准确率。

### 核心特性

- **分组管理**: 创建、编辑、删除热词分组
- **单选激活**: 同一时间只能激活一个分组，ASR 使用激活分组的热词
- **内联编辑**: 分组名称可直接点击编辑
- **热词管理**: 在分组内添加、删除热词
- **总开关**: 启用/禁用热词矫正功能

---

## 数据结构设计

### Config YAML 结构

```yaml
asr_term_correction:
  enabled: false
  active_group_id: "group-0"  # 当前激活的分组 ID
  groups:
    - id: "group-0"
      name: "默认分组"
      keyterms:
        - "FVG:fair value gap"
        - "POI:point of interest"
    - id: "group-1"
      name: "技术术语"
      keyterms:
        - "API:application programming interface"
        - "SDK:software development kit"
```

### 数据模型

**`api/models/schemas.py`**

```python
class HotwordGroup(BaseModel):
    """热词分组"""
    id: str = Field(..., description="分组唯一标识")
    name: str = Field(..., description="分组显示名称")
    keyterms: List[str] = Field(default_factory=list, description="热词列表")

class HotwordCorrectionConfig(BaseModel):
    """热词矫正配置"""
    enabled: bool = Field(default=False, description="是否启用")
    active_group_id: str = Field(default="group-0", description="激活分组 ID")
    groups: List[HotwordGroup] = Field(default_factory=list, description="分组列表")
```

### ID 生成策略

- 格式: `group-{index}`
- 示例: `group-0`, `group-1`, `group-2`
- 删除分组后**不重排**，保持原序号（简单高效）

---

## UI 设计

### 界面布局

```
┌─────────────────────────────────────────────┐
│ 热词矫正          [总开关]              │
├─────────────────────────────────────────────┤
│                                             │
│ [默认分组] [技术术语] [+ 新建分组]      │  ← 分组标签栏
│    (5)      (2)                          │    ← 显示热词数量
│                                             │
│ ┌─────────────────────────────────────────┐ │
│ │ FVG → fair value gap        [×]      │ │  ← 当前激活分组的热词列表
│ │ POI → point of interest      [×]      │ │
│ └─────────────────────────────────────────┘ │
│                                             │
│ [添加热词...]                    [+ 添加] │  ← 添加热词输入框
└─────────────────────────────────────────────┘
```

### 交互行为

| 操作 | 行为 |
|------|------|
| 点击分组标签 | 切换激活分组，显示该组热词 |
| 点击分组名称 | 进入内联编辑模式（输入框） |
| Enter / 失焦 | 保存编辑的分组名 |
| 点击删除按钮 | 删除分组（至少保留 1 个） |
| 点击 "+ 新建分组" | 显示输入框，输入名称创建 |

### 视觉样式

- **激活分组**: 蓝色背景 `bg-blue-100 text-blue-700`
- **未激活分组**: 灰色背景 `bg-gray-100 text-gray-700`
- **热词标签**: 复用现有圆角药丸样式 `rounded-lg`
- **分组标签**: 8px 圆角，flex 布局，hover 效果

---

## CSS 样式设计

**`webui/styles/components.css`**

```css
/* 分组标签栏 */
.hotword-groups-tabs {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  padding: 12px 0;
}

/* 分组标签 */
.hotword-group-tab {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 8px 12px;
  border-radius: 8px;
  cursor: pointer;
  transition: all 0.2s;
  background: #f5f5f7;
  color: #1d1d1f;
  font-size: 14px;
}

.hotword-group-tab.active {
  background: #e8f4fd;
  color: #0066cc;
}

.hotword-group-tab:hover {
  background: #e8e8ed;
}

.hotword-group-tab.active:hover {
  background: #d4e8f7;
}

/* 分组名称编辑输入框 */
.group-name-input {
  background: transparent;
  border: none;
  outline: none;
  font-size: 14px;
  width: 80px;
  font-weight: 500;
}

/* 分组操作按钮 */
.group-actions {
  display: flex;
  align-items: center;
  gap: 2px;
}

.group-action-btn {
  padding: 2px;
  border-radius: 4px;
  display: flex;
  align-items: center;
  justify-content: center;
}

.group-action-btn:hover {
  background: rgba(0, 0, 0, 0.08);
}
```

---

## 后端实现

### API 接口

**`GET /config`** - 返回完整配置
```json
{
  "asr_term_correction": {
    "enabled": true,
    "active_group_id": "group-0",
    "groups": [
      {"id": "group-0", "name": "默认分组", "keyterms": ["FVG:fair value gap"]},
      {"id": "group-1", "name": "技术术语", "keyterms": ["API:application"]}
    ]
  }
}
```

**`PUT /config`** - 保存完整配置（接收相同结构）

### 核心修改

**1. `core/utils/config_utils.py`**

```python
def init_hotword_groups(config):
    """初始化热词分组，如果没有 groups 则创建默认分组"""
    if "asr_term_correction" in config:
        atc = config["asr_term_correction"]
        if "groups" not in atc or not atc["groups"]:
            atc["groups"] = [{
                "id": "group-0",
                "name": "默认分组",
                "keyterms": []
            }]
            atc["active_group_id"] = "group-0"
    return config
```

**2. `core/_3_2_hotword.py`**

```python
def load_hotword_terms():
    """从配置中加载激活分组的热词"""
    enabled = load_key("asr_term_correction.enabled")
    if not enabled:
        return []

    active_group_id = load_key("asr_term_correction.active_group_id")
    groups = load_key("asr_term_correction.groups") or []

    # 找到激活分组
    active_group = next((g for g in groups if g["id"] == active_group_id), None)
    if active_group:
        return active_group.get("keyterms", [])
    return []
```

---

## 前端实现

### 数据结构

**`webui/js/components/settings.js`**

```javascript
data: {
  hotwordCorrection: {
    enabled: false,
    activeGroupId: 'group-0',
    groups: [
      { id: 'group-0', name: '默认分组', keyterms: [] }
    ]
  },
  nextGroupIndex: 1,
  editingGroupId: null,
  editingGroupName: '',
  newGroupName: ''
}
```

### 核心方法

| 方法 | 功能 |
|------|------|
| `activateGroup(groupId)` | 切换激活分组 |
| `addGroup(name)` | 创建新分组，激活新分组 |
| `deleteGroup(groupId)` | 删除分组（最少保留1个） |
| `startEditGroup(groupId, name)` | 进入编辑模式 |
| `saveEditGroup()` | 保存分组名 |
| `addHotword(text)` | 添加热词到当前激活分组 |
| `removeHotword(term)` | 删除热词 |
| `renderGroups()` | 渲染分组标签栏 |
| `renderHotwords()` | 渲染当前激活分组的热词 |

### 关键实现

**分组切换**
```javascript
activateGroup(groupId) {
  this.data.hotwordCorrection.activeGroupId = groupId;
  this.renderGroups();
  this.renderHotwords();
}
```

**添加分组**
```javascript
addGroup() {
  const newGroup = {
    id: `group-${this.data.nextGroupIndex++}`,
    name: this.data.newGroupName.trim(),
    keyterms: []
  };
  this.data.hotwordCorrection.groups.push(newGroup);
  this.data.newGroupName = '';
  this.data.hotwordCorrection.activeGroupId = newGroup.id;
  this.renderGroups();
  this.renderHotwords();
}
```

**删除分组**
```javascript
deleteGroup(groupId) {
  if (this.data.hotwordCorrection.groups.length <= 1) {
    showToast('至少保留一个分组', 'error');
    return;
  }
  const idx = this.data.hotwordCorrection.groups.findIndex(g => g.id === groupId);
  this.data.hotwordCorrection.groups.splice(idx, 1);

  // 如果删除的是激活分组，激活第一个
  if (this.data.hotwordCorrection.activeGroupId === groupId) {
    this.data.hotwordCorrection.activeGroupId = this.data.hotwordCorrection.groups[0].id;
  }
  this.renderGroups();
  this.renderHotwords();
}
```

**添加热词**（修改：添加到激活分组）
```javascript
addHotword() {
  const termInput = document.getElementById('newHotword');
  const input = termInput?.value.trim();
  if (!input) return;

  const activeGroup = this.data.hotwordCorrection.groups.find(
    g => g.id === this.data.hotwordCorrection.activeGroupId
  );

  if (activeGroup) {
    // 批量添加逻辑
    if (input.includes(',')) {
      const terms = input.split(',').map(t => t.trim()).filter(t => t.length > 0);
      activeGroup.keyterms.push(...terms);
      showToast(`已添加 ${terms.length} 个热词`, 'success');
    } else {
      activeGroup.keyterms.push(input);
      showToast('热词已添加', 'success');
    }
    this.renderHotwords();
  }

  if (termInput) termInput.value = '';
}
```

### 渲染方法

**`renderGroups()`** - 渲染分组标签栏
```javascript
renderGroups() {
  const container = document.getElementById('groupsTabs');
  const groups = this.data.hotwordCorrection.groups;
  const activeId = this.data.hotwordCorrection.activeGroupId;

  container.innerHTML = groups.map(group => {
    const isActive = group.id === activeId;
    const isEditing = this.data.editingGroupId === group.id;

    return `
      <div class="hotword-group-tab ${isActive ? 'active' : ''}"
           data-group-id="${group.id}"
           onclick="SettingsManager.activateGroup('${group.id}')">
        <!-- 图标 -->
        <svg>...</svg>
        <!-- 名称或编辑输入框 -->
        ${isEditing ? `
          <input type="text" class="group-name-input"
                 value="${this.data.editingGroupName}"
                 onkeydown="if(event.key==='Enter') SettingsManager.saveEditGroup()"
                 onblur="SettingsManager.saveEditGroup()"
                 onclick="event.stopPropagation()"
                 autofocus>
        ` : `
          <span class="group-name">${Utils.escapeHtml(group.name)}</span>
        `}
        <!-- 热词数量 -->
        <span class="group-count">(${group.keyterms.length})</span>
        <!-- 操作按钮 -->
        <div class="group-actions" onclick="event.stopPropagation()">
          ${!isEditing ? `
            <button onclick="SettingsManager.startEditGroup(...)">编辑</button>
            ${groups.length > 1 ? `
              <button onclick="SettingsManager.deleteGroup('${group.id}')">删除</button>
            ` : ''}
          ` : `
            <button onclick="SettingsManager.saveEditGroup()">保存</button>
          `}
        </div>
      </div>
    `;
  }).join('');

  // 新建分组按钮
  container.innerHTML += `
    <button class="hotword-group-tab" onclick="SettingsManager.showNewGroupInput()">
      <svg>...</svg>
      <span>新建分组</span>
    </button>
  `;
}
```

**`renderHotwords()`** - 渲染当前激活分组的热词
```javascript
renderHotwords() {
  const container = document.getElementById('hotwordsList');

  // 找到激活分组
  const activeGroup = this.data.hotwordCorrection.groups.find(
    g => g.id === this.data.hotwordCorrection.activeGroupId
  );

  if (!activeGroup || activeGroup.keyterms.length === 0) {
    container.innerHTML = '';
    return;
  }

  // 使用 activeGroup.keyterms 渲染热词（原有逻辑保持不变）
  container.innerHTML = activeGroup.keyterms.map((term) => {
    // ... 原有的热词渲染逻辑
  }).join('');
}
```

### 保存配置

```javascript
async saveSettings() {
  // ... 其他配置收集

  // 热词矫正配置（新结构）
  hotword_correction: {
    enabled: document.getElementById('hotwordEnabled')?.checked ?? this.data.hotwordCorrection.enabled,
    active_group_id: this.data.hotwordCorrection.activeGroupId,
    groups: this.data.hotwordCorrection.groups
  },

  // ... 调用 API 保存
}
```

---

## 修改文件清单

### 后端
- `api/models/schemas.py` - 新增 `HotwordGroup` 模型
- `api/routes/config.py` - 更新配置读写逻辑
- `core/utils/config_utils.py` - 添加默认分组初始化
- `core/_3_2_hotword.py` - 更正热词加载逻辑

### 前端
- `webui/js/components/settings.js` - 重构热词相关方法
- `webui/styles/components.css` - 新增分组标签样式

---

## 实施步骤

1. **后端数据模型** - 更新 schemas.py，添加 HotwordGroup
2. **后端配置逻辑** - 更新 config_utils.py 初始化逻辑
3. **后端 API** - 确保 /config 正确处理新结构
4. **后端热词加载** - 更新 _3_2_hotword.py 加载激活分组热词
5. **前端样式** - 添加分组标签 CSS
6. **前端数据结构** - 更新 settings.js 数据和方法
7. **前端渲染** - 实现分组标签和热词列表渲染
8. **测试** - 端到端测试分组创建、编辑、删除、热词管理

---

## 参考

- UI 交互逻辑参考：项目根目录 `KeytermGroupsSettings.tsx`
- 样式参考：现有 Apple 风格组件
- ASR 热词矫正流程：`core/_3_2_hotword.py`
