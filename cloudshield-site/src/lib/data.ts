import raw from '@/data/cloudshield.json'
import tonRaw from '@/data/toniot.json'

/**
 * Everything the site shows comes from two exported files:
 * - src/data/cloudshield.json (UNSW-NB15, regression track), written by scripts/export_data.py
 * - src/data/toniot.json (TON_IoT, classification and clustering), written by scripts/export_ton_iot.py
 */

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

/** One colour per UNSW-NB15 class (regression track). Normal is green; attacks run warm. */
export const CLASS_COLOR: Record<string, string> = {
  Normal: '#0f6e5a',
  Generic: '#2f5d9e',
  Exploits: '#e5481c',
  Fuzzers: '#c98a00',
  DoS: '#f08a24',
  Reconnaissance: '#7b3fa0',
  Analysis: '#8a7a2e',
  Backdoor: '#c2185b',
  Shellcode: '#2e7d32',
  Worms: '#4a4a4a',
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

export const CLUSTER_COLOR = ['#e5481c', '#2f5d9e', '#0f6e5a', '#c98a00', '#7b3fa0', '#c2185b', '#00838f', '#5b4636', '#8a7a2e', '#4a4a4a']

/* ------------------------------------------------------------------ TON_IoT (classification + clustering) */
export type TonFields = {
  proto: string; service: string; conn_state: string; duration: number
  src_bytes: number; dst_bytes: number; src_pkts: number; dst_pkts: number; dst_port: number
}
export type TonFlow = { id: number; label: string; pred: string; proba: { name: string; p: number }[]; f: TonFields }
export type TonModel = {
  model: string; key: string; part: 'A' | 'B'; tuned: boolean
  acc: number; f1w: number; f1m: number; params: string
}
export type TuneRow = { n_estimators: number; learning_rate: number; max_depth: number; cv: number; cvStd: number; test: number; testAcc: number }

export type TonData = {
  overview: {
    name: string; file: string; source: string; citation: string; url: string
    rows: number; rowsClean: number; duplicates: number; columns: number; machines: number; attackShare: number
    classes: { name: string; count: number; machines: number }[]
  }
  /** From notebooks/03_classification_1.ipynb: test set = machines never seen in training */
  classification: {
    best: string
    bestDisplay: string
    split: string
    /** classes present in the test fold, most frequent first */
    order: string[]
    allClasses: string[]
    missingFromTest: string[]
    models: TonModel[]
    final: { acc: number; f1w: number; f1m: number; precision: number; recall: number; auc: number }
    finalParams: Record<string, number>
    searches: { model: string; params: string; cv: number; test: number }[]
    perClass: { name: string; precision: number; recall: number; f1: number }[]
    confusion: { rows: string[]; cols: string[]; counts: number[][]; norm: number[][] }
    recall: Record<string, number>
    trainRows: number; testRows: number; features: number; trainMachines: number; testMachines: number
    lab: TonFlow[]
    stream: TonFlow[]
    /** [P(attack) from the final model, class index in `allClasses`]; uncertain flows oversampled */
    firewall: { flows: [number, number][]; hardShare: number; unsureShareTest: number; oversampled: boolean }
    /** 5-fold CV grouped by source machine, training set only */
    cv5: { model: string; mean: number; std: number; f1m: number; acc: number; folds: number[] }[]
    tuning: { sampleRows: number; grid: TuneRow[] } | null
  }
  clustering: {
    sample: number; k: number
    elbow: { k: number; inertia: number; silhouette: number }[]
    linkages: { linkage: string; silhouette: number; largest: number }[]
    metrics: { name: string; silhouette: number; daviesBouldin: number; calinskiHarabasz: number }[]
    stability: { silhouette: [number, number]; ari: [number, number] | null }
    pcaVariance: number[]
    clusters: { id: number; size: number; mix: { name: string; share: number }[] }[]
    points: [number, number, number, number][]
  }
}

export const T = tonRaw as unknown as TonData

/** One colour per TON_IoT class. Normal is green; attacks are distinct inks that print well on paper. */
export const TON_COLOR: Record<string, string> = {
  normal: '#0f6e5a',
  scanning: '#2f5d9e',
  ddos: '#e5481c',
  dos: '#f08a24',
  injection: '#7b3fa0',
  password: '#c2185b',
  backdoor: '#5b4636',
  xss: '#b8930a',
  ransomware: '#16140f',
  mitm: '#00838f',
}

export const TON_NOTE: Record<string, string> = {
  normal: 'Ordinary IoT and IT traffic on the testbed.',
  scanning: 'Probing hosts and ports to map the network before an attack.',
  ddos: 'Many machines flooding one target until it stops answering.',
  dos: 'One machine flooding a target to exhaust its resources.',
  injection: 'Smuggling commands or SQL into an input a service trusts.',
  password: 'Guessing or cracking credentials, for example by brute force.',
  backdoor: 'A hidden channel left behind for remote access later.',
  xss: 'Cross-site scripting: injecting script into a web page other users load.',
  ransomware: 'Malware that encrypts files and demands payment.',
  mitm: 'Man-in-the-middle: secretly relaying or altering traffic between two parties.',
}

export const tonLabel = (c: string) => (c === 'ddos' ? 'DDoS' : c === 'dos' ? 'DoS' : c === 'xss' ? 'XSS' : c === 'mitm' ? 'MITM' : c[0].toUpperCase() + c.slice(1))
