import { PageLayout, SharedLayout } from "./quartz/cfg"
import * as Component from "./quartz/components"

// components shared across all pages
export const sharedPageComponents: SharedLayout = {
  head: Component.Head(),
  header: [],
  // Breadcrumb at the very bottom of every content page
  afterBody: [Component.ReturnHome()],
  footer: Component.Footer({
    links: {
      // Add your own links here if desired, or leave empty
    },
  }),
}

// components for pages that display a single page (e.g. a single note)
export const defaultContentPageLayout: PageLayout = {
  // Title followed by a breadcrumb link back to home
  beforeBody: [
    Component.ArticleTitle(),
    Component.ReturnHome(),
    // Removed: ContentMeta, TagList
  ],
  left: [
    // Completely empty - no sidebar
  ],
  right: [
    // Completely empty - no Graph, TableOfContents, or Backlinks
  ],
}

// components for pages that display lists of pages  (e.g. tags or folders)
export const defaultListPageLayout: PageLayout = {
  beforeBody: [Component.Breadcrumbs(), Component.ArticleTitle(), Component.ContentMeta()],
  left: [
    Component.PageTitle(),
    Component.MobileOnly(Component.Spacer()),
    Component.Flex({
      components: [
        {
          Component: Component.Search(),
          grow: true,
        },
        { Component: Component.Darkmode() },
      ],
    }),
    Component.Explorer(),
  ],
  right: [],
}
