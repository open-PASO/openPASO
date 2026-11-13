import 'katex/dist/katex.min.css'
import ReactMarkdown from 'react-markdown'
import rehypeKatex from 'rehype-katex'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import { fileUrl, tidy } from '../api'

/** A picture a message names, as a URL inside this run's work folder; anything else is not loaded. */
function runMedia(src: string | undefined, runId?: string): string | null {
  if (!src || !runId) return null
  const marker = `/webui_${runId}/work/`
  const at = src.replace(/\\/g, '/').indexOf(marker)
  const rel = at >= 0 ? src.replace(/\\/g, '/').slice(at + marker.length)
    : /^[a-z][a-z0-9+.-]*:|^[/\\~]/i.test(src) ? null : src.replace(/^\.\//, '')
  if (rel == null || rel.split('/').includes('..')) return null
  return fileUrl(`webui_${runId}/work/${rel}`)
}

/* Models answer in markdown with LaTeX maths: lists, tables, code, $…$ and $$…$$.
   Formulas used to print as source, and a "\n" inside "\nabla" was turned into a
   line break on the way. */
export default function Markdown({ text, className = '', runId }: { text: string; className?: string; runId?: string }) {
  // models also write \( … \) and \[ … \]; remark-math reads dollars only
  const src = tidy(text)
    .replace(/\\\[([\s\S]+?)\\\]/g, (_, m) => `\n$$\n${m}\n$$\n`)
    .replace(/\\\(([\s\S]+?)\\\)/g, (_, m) => `$${m}$`)
  return (
    <div className={`md ${className}`}>
      <ReactMarkdown remarkPlugins={[remarkGfm, remarkMath]}
                     rehypePlugins={[[rehypeKatex, { throwOnError: false, strict: 'ignore' }]]}
                     components={{
                       a: (p) => <a {...p} target="_blank" rel="noreferrer" />,
                       img: ({ src, alt }) => {
                         const url = runMedia(typeof src === 'string' ? src : undefined, runId)
                         if (!url) return <span className="text-muted">[{alt || 'picture'}: not a file of this run]</span>
                         return /\.(mp4|webm)$/i.test(url)
                           ? <video src={url} controls loop muted playsInline className="max-w-full rounded-[6px] bg-black my-2" />
                           : <img src={url} alt={alt || ''} loading="lazy" className="max-w-full rounded-[6px] bg-white my-2" />
                       },
                     }}>
        {src}
      </ReactMarkdown>
    </div>
  )
}
