// API 封装 —— 字段契约与后端 src/api/models.py 对齐（R3）
import axios from 'axios'

const http = axios.create({
  baseURL: '/api',
  timeout: 185000 // 略大于后端超时（ReAct 多跳链路可达分钟级）
})

/** 同步问诊（流式失败时的兜底路径） */
export async function consult(query, sessionId) {
  const { data } = await http.post('/consult', { query, session_id: sessionId })
  return data
}

/**
 * 流式问诊（POST SSE；EventSource 不支持 POST，故用 fetch + ReadableStream 手工解析帧）。
 * 事件协议：progress{node,label,facts} → answer{完整JSON} → error{message} → done [DONE]
 */
export async function consultStream(query, sessionId, handlers = {}) {
  const { onProgress, onAnswer, onError, onDone } = handlers
  const resp = await fetch('/api/consult/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json; charset=utf-8' },
    body: JSON.stringify({ query, session_id: sessionId })
  })
  if (!resp.ok || !resp.body) throw new Error(`HTTP ${resp.status}`)

  const reader = resp.body.getReader()
  const decoder = new TextDecoder('utf-8')
  let buffer = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })

    // SSE 帧以空行分隔
    let sep
    while ((sep = buffer.indexOf('\n\n')) !== -1) {
      const frame = buffer.slice(0, sep)
      buffer = buffer.slice(sep + 2)
      let event = 'message'
      let data = ''
      for (const line of frame.split('\n')) {
        if (line.startsWith('event: ')) event = line.slice(7).trim()
        else if (line.startsWith('data: ')) data += line.slice(6)
      }
      if (event === 'progress') onProgress && onProgress(JSON.parse(data))
      else if (event === 'answer') onAnswer && onAnswer(JSON.parse(data))
      else if (event === 'error') onError && onError(JSON.parse(data))
      else if (event === 'done' && data.trim() === '[DONE]') {
        onDone && onDone()
        return
      }
    }
  }
  onDone && onDone()
}

export async function health() {
  const { data } = await http.get('/health')
  return data
}

export async function getStats() {
  const { data } = await http.get('/stats')
  return data
}

export async function getConversations() {
  const { data } = await http.get('/conversations')
  return data
}

export async function getConversation(id) {
  const { data } = await http.get(`/conversations/${id}`)
  return data
}

export async function deleteConversation(id) {
  const { data } = await http.delete(`/conversations/${id}`)
  return data
}
