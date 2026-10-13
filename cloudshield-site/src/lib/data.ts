import raw from '@/data/cloudshield.json'

/** Everything the site shows comes from src/data/cloudshield.json, written by scripts/export_data.py. */

export type FlowFields = {
  proto: string; service: string; state: string
  dur: number; spkts: number; dpkts: number; sbytes: number; dbytes: number
  sttl: number; dttl: number; ct_srv_src: number
}

export type Flow = {
  id: number
  label: string
  pred: string
  proba: { name: string; p: number }[]
  f: FlowFields
}

export type ForecastFlow = {
  id: number
  label: string
  f: FlowFields
  actual: number
  pred: number
  lo: number
  hi: number
  whatIfAt: number
  /** model output at each what-if value: [mean, p10, p90] in bytes */
  sweep: [number, number, number][]
}

export type CsData = {
  overview: {
    dataset: { name: string; source: string; citation: string; url: string }
    rows: number; features: number; classes: number; attackShare: number
    duplicates: number; duplicateShare: number; contamination: number; officialOverlap: number; imbalance: number
    classCounts: { name: string; count: number }[]
    protocols: number; services: number
  }
  classification: {
    best: string
    k: number
    order: string[]
    models: { model: string; accVal: number; acc: number; f1w: number; f1m: number; fit: number }[]
    baseline: { acc: number; note: string }
    cv: { model: string; mean: number; std: number }[]
    perClass: { name: string; precision: number; recall: number; f1: number }[]
    confusion: { labels: string[]; norm: number[][]; counts: number[][] }
    testRows: number
    trainRows: number
    lab: Flow[]
    stream: Flow[]
  }
  regression: {
    target: string
    best: string
    models: { model: string; r2Val: number; r2: number; rmseLog: number; maeLog: number; rmse: number; mae: number; fit: number }[]
    tuning: { model: string; r2Before: number; r2After: number; rmseBefore: number; rmseAfter: number }[]
    cv: { model: string; mean: number; std: number }[]
    baselineRmse: number
    importance: { name: string; value: number }[]
    scatter: [number, number][]
    leak: { columns: string[]; corr: Record<string, number>; rebuild: number }
    whatIf: { feature: string; values: number[] }
    flows: ForecastFlow[]
    trainRows: number
    features: number
  }
  clustering: {
    note: string
    sample: number
    k: number
    elbow: { k: number; inertia: number; silhouette: number }[]
    kmeans: { silhouette: number; daviesBouldin: number; calinskiHarabasz: number }
    agglomerative: { silhouette: number; daviesBouldin: number; calinskiHarabasz: number }
    pcaVariance: number[]
    clusters: { id: number; size: number; mix: { name: string; share: number }[] }[]
    /** [x, y, cluster, class index in classification.order], x and y scaled to 0..1 */
    points: [number, number, number, number][]
  }
}

export const D = raw as unknown as CsData

export const pct = (v: number | null | undefined, digits = 0) =>
  v == null || Number.isNaN(v) ? '–' : `${(100 * v).toFixed(digits)}%`

export const int = (v: number) => Math.round(v).toLocaleString('en-US')

/** 1234567 -> "1.23 MB" (decimal units, as network tools report them). */
export function bytes(v: number, digits = 1) {
  if (v < 1000) return `${Math.round(v)} B`
  const units = ['KB', 'MB', 'GB']
  let x = v / 1000
  let i = 0
  while (x >= 1000 && i < units.length - 1) {
    x /= 1000
    i++
  }
  return `${x.toFixed(x < 10 ? digits + 1 : digits)} ${units[i]}`
}

/** One colour per traffic class. Normal is the network's own cyan; attacks run warm. */
export const CLASS_COLOR: Record<string, string> = {
  Normal: '#36e2d4',
  Generic: '#7aa2ff',
  Exploits: '#ff3b5c',
  Fuzzers: '#ffb547',
  DoS: '#ff7a3d',
  Reconnaissance: '#c38bff',
  Analysis: '#f2e46b',
  Backdoor: '#ff6fb5',
  Shellcode: '#5fd38d',
  Worms: '#f4f4f4',
}

export const CLASS_NOTE: Record<string, string> = {
  Normal: 'Ordinary traffic: web, mail, file transfer, DNS.',
  Generic: 'Attacks that work against any block cipher, regardless of its structure.',
  Exploits: 'The attacker knows a security hole in an operating system or service and uses it.',
  Fuzzers: 'Floods a program or network with random data to make it crash.',
  DoS: 'Denial of service: makes a server or network unavailable to its users.',
  Reconnaissance: 'Scans and probes that gather information before an attack.',
  Analysis: 'Port scans, spam and HTML-file penetration attacks.',
  Backdoor: 'Bypasses authentication to get unauthorised remote access.',
  Shellcode: 'A small piece of code used as the payload of an exploit.',
  Worms: 'Copies itself to spread to other computers.',
}

export const CLUSTER_COLOR = ['#36e2d4', '#7aa2ff', '#ffb547', '#ff3b5c', '#c38bff', '#5fd38d', '#ff6fb5', '#f2e46b', '#ff7a3d', '#f4f4f4']
