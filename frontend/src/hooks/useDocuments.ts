import { useCallback, useState } from 'react'
import { deleteDocument, errorText, listDocuments, uploadDocument } from '../api.ts'
import { demoDocuments } from '../demo.ts'
import { checkUpload } from '../lib/format.ts'
import type { DocumentItem } from '../types.ts'

const WAIT_NOTICE_MS = 4000

export type DocumentsState = ReturnType<typeof useDocuments>

export function useDocuments(demo: boolean) {
  const [documents, setDocuments] = useState<DocumentItem[]>([])
  const [loaded, setLoaded] = useState(false)
  const [loadError, setLoadError] = useState('')
  const [uploading, setUploading] = useState<string | null>(null)
  const [uploadError, setUploadError] = useState('')
  const [deleting, setDeleting] = useState<string | null>(null)
  const [deleteError, setDeleteError] = useState('')

  const reload = useCallback(async () => {
    try {
      setDocuments(await listDocuments())
      setLoaded(true)
      setLoadError('')
    } catch (error) {
      setLoadError(errorText(error))
    }
  }, [])

  const upload = useCallback(
    async (files: File[], maxUploadMb: number, streaming: boolean) => {
      if (files.length === 0) return
      const [file] = files
      const wait = uploading
        ? `Wait until ${uploading} has finished indexing.`
        : deleting
          ? 'Wait until the delete has finished.'
          : streaming
            ? 'Wait until the answer has finished.'
            : ''
      const problem = demo
        ? 'Turn off demo data to upload a file.'
        : wait || (files.length > 1 ? 'Upload one file at a time.' : checkUpload(file.name, file.size, maxUploadMb))
      setUploadError(problem)
      // A wait notice stops being true on its own, so it must not outlive the wait.
      if (wait) setTimeout(() => setUploadError((current) => (current === wait ? '' : current)), WAIT_NOTICE_MS)
      if (problem) return

      setUploading(file.name)
      try {
        await uploadDocument(file)
        await reload()
      } catch (error) {
        setUploadError(errorText(error))
      } finally {
        setUploading(null)
      }
    },
    [demo, uploading, deleting, reload],
  )

  const remove = useCallback(
    async (filename: string) => {
      setDeleting(filename)
      setDeleteError('')
      try {
        await deleteDocument(filename)
      } catch (error) {
        setDeleteError(errorText(error))
      }
      // The server can report a failure after the file is gone, so always re-read the list.
      await reload()
      setDeleting(null)
    },
    [reload],
  )

  const state = { documents, loaded, loadError, uploading, uploadError, deleting, deleteError, reload, upload, remove }
  return demo ? { ...state, documents: demoDocuments, loaded: true, loadError: '', deleteError: '' } : state
}
