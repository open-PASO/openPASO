/** Open a run's result file in the 3D view; the run page listens for this. */
export function open3d(rel: string) {
  window.dispatchEvent(new CustomEvent('openpaso:view3d', { detail: { rel } }))
}

export const RESULT_3D = /\.(vtu|vtk|vtp|pvtu|pvd)$/i
