import { QuartzComponent, QuartzComponentConstructor } from "./types"

export default (() => {
  const ReturnHome: QuartzComponent = () => (
    <nav class="breadcrumb-container" aria-label="breadcrumbs">
      <a href="/index.html">← Return to Home</a>
    </nav>
  )

  return ReturnHome
}) satisfies QuartzComponentConstructor
