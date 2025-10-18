import { QuartzConfig } from "./quartz/cfg"
import * as Plugin from "./quartz/plugins"

/**
 * Quartz 4 Configuration
 * Docs: https://quartz.jzhao.xyz/configuration
 */
const config: QuartzConfig = {
  configuration: {
    pageTitle: "Michael Zeng",
    pageTitleSuffix: "",
    enableSPA: true,
    enablePopovers: true,
    analytics: { provider: "plausible" },
    locale: "en-US",
    // Keep your existing baseUrl
    baseUrl: "michaelszeng.github.io",
    ignorePatterns: ["private", "templates", ".obsidian"],
    defaultDateType: "modified",
    theme: {
      fontOrigin: "googleFonts",
      cdnCaching: true,
      typography: {
        header: "Schibsted Grotesk",
        body: "Source Sans Pro",
        code: "IBM Plex Mono",
      },
      colors: {
        lightMode: {
          light: "#faf8f8",
          lightgray: "#e5e5e5",
          gray: "#b8b8b8",
          darkgray: "#4e4e4e",
          dark: "#2b2b2b",
          secondary: "#284b63",
          tertiary: "#84a59d",
          highlight: "rgba(143, 159, 169, 0.15)",
          textHighlight: "#fff23688",
        },
        darkMode: {
          light: "#161618",
          lightgray: "#393639",
          gray: "#646464",
          darkgray: "#d4d4d4",
          dark: "#ebebec",
          secondary: "#7b97aa",
          tertiary: "#84a59d",
          highlight: "rgba(143, 159, 169, 0.15)",
          textHighlight: "#b3aa0288",
        },
      },
    },
  },

  plugins: {
    transformers: [
      Plugin.FrontMatter(),
      Plugin.CreatedModifiedDate({ priority: ["frontmatter", "git", "filesystem"] }),
      Plugin.SyntaxHighlighting({
        theme: { light: "github-light", dark: "github-dark" },
        keepBackground: false,
      }),

      // Keep OFM and GFM, but put LaTeX RIGHT AFTER them so math is parsed
      // before other transformers (TOC, Description, CrawlLinks, etc.).
      Plugin.ObsidianFlavoredMarkdown({ enableInHtmlEmbed: false }),
      Plugin.GitHubFlavoredMarkdown(),

      // === CRITICAL: Math ===
      // Use MathJax (closest to Obsidian). Provide AMS + macros + sane delimiters.
      Plugin.Latex({
        renderEngine: "mathjax",
        options: {
          tex: {
            inlineMath: [["$", "$"], ["\\(", "\\)"]],
            displayMath: [["$$", "$$"]],
            processEscapes: true,
            processEnvironments: true,
            tags: "ams",               // enable align/align* numbering rules
            // Add the macros you use frequently so you don't have to rewrite notes
            macros: {
              "\\argmin": "\\operatorname*{arg\\,min}",
              "\\argmax": "\\operatorname*{arg\\,max}",
              "\\E": "\\mathbb{E}",
              "\\KL": "\\mathrm{D}_{\\mathrm{KL}}",
            },
            // Be permissive—Obsidian is forgiving; mirror that leniency
            strict: false,
          },
          svg: { fontCache: "global" },  // crisp math, no client-side render
        },
      }),

      // The rest of your usual transformers
      Plugin.TableOfContents(),
      Plugin.CrawlLinks({ markdownLinkResolution: "shortest" }),
      Plugin.Description(),
    ],

    filters: [Plugin.RemoveDrafts()],

    emitters: [
      Plugin.AliasRedirects(),
      Plugin.ComponentResources(), // ensures CSS/JS (including math assets) land in output
      Plugin.ContentPage(),
      Plugin.FolderPage(),
      Plugin.TagPage(),
      Plugin.ContentIndex({ enableSiteMap: true, enableRSS: true }),
      Plugin.Assets(),
      Plugin.Static(),
      Plugin.Favicon(),
      Plugin.NotFoundPage(),
      Plugin.CustomOgImages(), // keep as you had it
    ],
  },
}

export default config
