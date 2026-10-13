import { gsap } from 'gsap'
import { ScrollTrigger } from 'gsap/ScrollTrigger'
import { SplitText } from 'gsap/SplitText'
import { ScrambleTextPlugin } from 'gsap/ScrambleTextPlugin'
import { DrawSVGPlugin } from 'gsap/DrawSVGPlugin'
import { useGSAP } from '@gsap/react'

gsap.registerPlugin(useGSAP, ScrollTrigger, SplitText, ScrambleTextPlugin, DrawSVGPlugin)
gsap.defaults({ ease: 'power3.out', duration: 0.9 })

export const prefersReducedMotion = () =>
  typeof window !== 'undefined' && window.matchMedia('(prefers-reduced-motion: reduce)').matches

export { gsap, ScrollTrigger, SplitText, useGSAP }
