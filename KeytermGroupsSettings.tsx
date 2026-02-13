import React, { useState } from 'react';
import { FolderOpen, Plus, X, Trash2, Edit2, Check } from 'lucide-react';
import type { KeytermGroup } from '@/types/transcription';

interface KeytermGroupsSettingsProps {
  groups: KeytermGroup[];
  onGroupsChange: (groups: KeytermGroup[]) => void;
  keytermsEnabled: boolean;
  onKeytermsEnabledChange: (enabled: boolean) => void;
}

export const KeytermGroupsSettings: React.FC<KeytermGroupsSettingsProps> = ({
  groups,
  onGroupsChange,
  keytermsEnabled,
  onKeytermsEnabledChange
}) => {
  const [activeGroupId, setActiveGroupId] = useState(groups[0]?.id || '');
  const [newKeyterm, setNewKeyterm] = useState('');
  const [newGroupName, setNewGroupName] = useState('');
  const [editingGroupId, setEditingGroupId] = useState<string | null>(null);
  const [editingName, setEditingName] = useState('');

  const activeGroup = groups.find(g => g.id === activeGroupId);

  const addGroup = () => {
    if (newGroupName.trim()) {
      const newGroup: KeytermGroup = {
        id: `group-${Date.now()}`,
        name: newGroupName.trim(),
        keyterms: []
      };
      onGroupsChange([...groups, newGroup]);
      setNewGroupName('');
      setActiveGroupId(newGroup.id);
    }
  };

  const deleteGroup = (groupId: string) => {
    onGroupsChange(groups.filter(g => g.id !== groupId));
    if (activeGroupId === groupId) {
      setActiveGroupId(groups[0]?.id || '');
    }
  };

  const startEditGroup = (group: KeytermGroup) => {
    setEditingGroupId(group.id);
    setEditingName(group.name);
  };

  const saveEditGroup = () => {
    if (editingName.trim()) {
      onGroupsChange(groups.map(g =>
        g.id === editingGroupId ? { ...g, name: editingName.trim() } : g
      ));
      setEditingGroupId(null);
      setEditingName('');
    }
  };

  const addKeyterm = () => {
    if (activeGroup && newKeyterm.trim() && !activeGroup.keyterms.includes(newKeyterm.trim())) {
      onGroupsChange(groups.map(g =>
        g.id === activeGroupId
          ? { ...g, keyterms: [...g.keyterms, newKeyterm.trim()] }
          : g
      ));
      setNewKeyterm('');
    }
  };

  const removeKeyterm = (term: string) => {
    onGroupsChange(groups.map(g =>
      g.id === activeGroupId
        ? { ...g, keyterms: g.keyterms.filter(k => k !== term) }
        : g
    ));
  };

  return (
    <div className="space-y-6">
      {/* 热词分组 */}
      <div className="space-y-3">
        <div className="flex justify-between items-center">
          <h3 className="apple-heading-small">热词提示</h3>
          <button
            onClick={() => onKeytermsEnabledChange(!keytermsEnabled)}
            className={`w-12 h-6 rounded-full transition-colors ${
              keytermsEnabled ? 'bg-blue-500' : 'bg-gray-300'
            }`}
          >
            <div className={`w-5 h-5 bg-white rounded-full shadow transition-transform ${
              keytermsEnabled ? 'translate-x-6' : 'translate-x-0.5'
            }`} />
          </button>
        </div>

        {/* 分组标签 */}
        <div className="flex flex-wrap gap-2">
          {groups.map((group) => (
            <div
              key={group.id}
              className={`flex items-center gap-2 px-3 py-2 rounded-lg cursor-pointer transition-all ${
                activeGroupId === group.id
                  ? 'bg-blue-100 text-blue-700'
                  : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
              }`}
              onClick={() => setActiveGroupId(group.id)}
            >
              <FolderOpen className="h-4 w-4" />
              {editingGroupId === group.id ? (
                <input
                  type="text"
                  value={editingName}
                  onChange={(e) => setEditingName(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && saveEditGroup()}
                  onClick={(e) => e.stopPropagation()}
                  className="bg-transparent border-none outline-none text-sm w-24"
                  autoFocus
                />
              ) : (
                <span className="text-sm font-medium">{group.name}</span>
              )}
              <span className="text-xs opacity-60">({group.keyterms.length})</span>
              {editingGroupId === group.id ? (
                <button
                  onClick={(e) => { e.stopPropagation(); saveEditGroup(); }}
                  className="hover:bg-blue-200 rounded p-0.5"
                >
                  <Check className="h-3 w-3" />
                </button>
              ) : (
                <div className="flex items-center gap-0.5">
                  <button
                    onClick={(e) => { e.stopPropagation(); startEditGroup(group); }}
                    className="hover:bg-blue-200 rounded p-0.5"
                  >
                    <Edit2 className="h-3 w-3" />
                  </button>
                  {groups.length > 1 && (
                    <button
                      onClick={(e) => { e.stopPropagation(); deleteGroup(group.id); }}
                      className="hover:bg-blue-200 rounded p-0.5"
                    >
                      <Trash2 className="h-3 w-3" />
                    </button>
                  )}
                </div>
              )}
            </div>
          ))}

          {/* 新建分组输入框 */}
          {newGroupName ? (
            <div className="flex items-center gap-2 px-3 py-2 bg-gray-50 rounded-lg">
              <input
                type="text"
                value={newGroupName}
                onChange={(e) => setNewGroupName(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && addGroup()}
                onBlur={() => {
                  if (newGroupName.trim()) addGroup();
                  else setNewGroupName('');
                }}
                placeholder="分组名称"
                className="bg-transparent border-none outline-none text-sm w-24"
                autoFocus
              />
              <button
                onClick={addGroup}
                className="text-green-600 hover:text-green-700"
              >
                <Check className="h-4 w-4" />
              </button>
            </div>
          ) : (
            <button
              onClick={() => setNewGroupName('新建分组')}
              className="flex items-center gap-2 px-3 py-2 bg-gray-50 text-gray-600 rounded-lg hover:bg-gray-200 transition-colors"
            >
              <Plus className="h-4 w-4" />
              <span className="text-sm">新建分组</span>
            </button>
          )}
        </div>

        {/* 当前分组的热词列表 */}
        {activeGroup && (
          <div className="space-y-3">
            {activeGroup.keyterms.length > 0 && (
              <div className="flex flex-wrap gap-2">
                {activeGroup.keyterms.map((term) => (
                  <div
                    key={term}
                    className="flex items-center gap-1 px-3 py-1.5 bg-gray-100 rounded-lg text-sm text-gray-700"
                  >
                    <span>{term}</span>
                    <button
                      onClick={() => removeKeyterm(term)}
                      className="hover:text-gray-900"
                    >
                      <X className="h-3.5 w-3.5" />
                    </button>
                  </div>
                ))}
              </div>
            )}

            {/* 添加热词 */}
            <div className="flex gap-2">
              <input
                type="text"
                value={newKeyterm}
                onChange={(e) => setNewKeyterm(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && addKeyterm()}
                placeholder="添加热词..."
                className="flex-1 p-3 bg-gray-50 border border-gray-200 rounded-lg text-gray-900 focus:outline-none focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20 transition-all"
              />
              <button
                onClick={addKeyterm}
                className="apple-button apple-button-secondary"
              >
                <Plus className="h-4 w-4" />
                <span>添加</span>
              </button>
            </div>
          </div>
        )}
      </div>

      {/* 说明信息 */}
      <div className="bg-blue-50 border border-blue-200 rounded-xl p-4">
        <div className="flex items-start gap-3">
          <FolderOpen className="h-5 w-5 text-blue-600 flex-shrink-0 mt-0.5" />
          <div className="text-sm text-blue-700">
            <p className="font-medium mb-1">热词提示说明</p>
            <p className="text-blue-600">
              按领域分组管理热词，提高专业术语识别准确率。所有分组的词汇将一起发送给 ASR 服务。
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};
