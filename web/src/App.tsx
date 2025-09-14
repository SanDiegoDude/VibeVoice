import { useEffect, useMemo, useState } from 'react'

type Model = { id: string; display_name: string }

export default function App() {
  const [models, setModels] = useState<Model[]>([])
  const [selectedModel, setSelectedModel] = useState<string>('')
  const [showTemplatesModal, setShowTemplatesModal] = useState(false)

  useEffect(() => {
    fetch('http://localhost:8081/models')
      .then(r => r.json())
      .then((data: Model[]) => {
        setModels(data)
        if (data.length > 0) setSelectedModel(data[0].id)
      })
      .catch(() => {})
  }, [])

  return (
    <div className="h-full grid grid-cols-12 gap-4 p-4">
      {/* Left: 1/3 speakers & voice settings */}
      <section className="col-span-12 md:col-span-4 space-y-4">
        <div className="p-4 rounded-lg bg-white shadow border">
          <h2 className="font-semibold mb-2">Speakers & Voice Settings</h2>
          <div className="text-sm text-slate-600">Coming soon</div>
        </div>
      </section>

      {/* Center: 2/3 chat + top row controls */}
      <section className="col-span-12 md:col-span-8 space-y-4">
        <div className="p-4 rounded-lg bg-white shadow border flex items-center gap-2 flex-wrap">
          <select
            className="border rounded px-2 py-1"
            value={selectedModel}
            onChange={e => setSelectedModel(e.target.value)}
          >
            {models.map(m => (
              <option key={m.id} value={m.id}>{m.display_name}</option>
            ))}
          </select>
          <button className="px-3 py-1 rounded bg-slate-800 text-white">Start New Conversation</button>
          <div className="ml-auto flex gap-2">
            <button className="px-3 py-1 rounded border" onClick={() => setShowTemplatesModal(true)}>Language Config</button>
          </div>
        </div>

        <div className="grid grid-cols-12 gap-4">
          {/* Chat (left of center area) */}
          <div className="col-span-12 lg:col-span-7 p-4 rounded-lg bg-white shadow border min-h-[400px]">
            <h2 className="font-semibold mb-2">Chat</h2>
            <div className="text-sm text-slate-600">Agent chat and prompts go here</div>
          </div>
          {/* Right column inside center: VibeVoice audio panel */}
          <div className="col-span-12 lg:col-span-5 p-4 rounded-lg bg-white shadow border min-h-[400px]">
            <h2 className="font-semibold mb-2">VibeVoice Audio</h2>
            <div className="space-y-2">
              <textarea className="w-full h-40 border rounded p-2" placeholder="Conversation script here..." />
              <button className="px-3 py-1 rounded bg-emerald-600 text-white">Generate Audio</button>
              <div className="text-sm text-slate-600">Streaming player & waveform here</div>
            </div>
          </div>
        </div>
      </section>

      {showTemplatesModal && (
        <div className="fixed inset-0 bg-black/30 flex items-center justify-center">
          <div className="bg-white p-4 rounded-lg shadow w-full max-w-3xl">
            <div className="flex items-center justify-between mb-2">
              <h3 className="font-semibold">Language Templates</h3>
              <button className="px-2 py-1 border rounded" onClick={() => setShowTemplatesModal(false)}>Close</button>
            </div>
            <div className="text-sm text-slate-600">Template manager UI will go here (monologue/dialogue, versions)</div>
          </div>
        </div>
      )}
    </div>
  )
}

