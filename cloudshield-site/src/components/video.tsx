import { useEffect, useRef } from 'react'
import { cn } from '@/lib/utils'

/**
 * Motion graphics from Vecteezy (free licence, credited in the footer), re-encoded to
 * 1280px H.264 and stored in public/videos so the demo needs no internet connection.
 */
export const VIDEOS = {
  shield: { src: '/videos/shield.mp4', poster: '/videos/shield.jpg', title: 'A shield with a keyhole drawn in cyan particles' },
  glitch: { src: '/videos/glitch.mp4', poster: '/videos/glitch.jpg', title: 'Red glitching text reading System error' },
  attackMap: { src: '/videos/attack-map.mp4', poster: '/videos/attack-map.jpg', title: 'A world map on a monitor with cyber-attack targets lighting up' },
  switch: { src: '/videos/switch.mp4', poster: '/videos/switch.jpg', title: 'Fibre-optic cables plugged into a network switch, status lights blinking' },
  fibre: { src: '/videos/fibre.mp4', poster: '/videos/fibre.jpg', title: 'Strands of optical fibre bursting outward in teal light' },
  serverRoom: { src: '/videos/server-room.mp4', poster: '/videos/server-room.jpg', title: 'Moving down a data-centre corridor between server racks' },
} as const

export type VideoKey = keyof typeof VIDEOS

type Props = {
  video: VideoKey
  className?: string
  /** Fill the parent like object-fit: cover instead of keeping a 16:9 box. */
  cover?: boolean
}

/** Muted, looping background video that only plays while it is on screen. */
export function Video({ video, className, cover }: Props) {
  const ref = useRef<HTMLVideoElement>(null)
  const v = VIDEOS[video]

  useEffect(() => {
    const el = ref.current!
    const io = new IntersectionObserver(([e]) => {
      if (e.isIntersecting) el.play().catch(() => {})
      else el.pause()
    }, { rootMargin: '200px' })
    io.observe(el)
    return () => io.disconnect()
  }, [])

  const media = (
    <video
      ref={ref}
      src={v.src}
      poster={v.poster}
      aria-label={v.title}
      className="pointer-events-none absolute inset-0 h-full w-full object-cover"
      muted
      loop
      playsInline
      preload="metadata"
    />
  )
  if (cover) return <div className={cn('absolute inset-0 overflow-hidden', className)}>{media}</div>
  return (
    <div className={cn('relative w-full overflow-hidden', className)} style={{ aspectRatio: '16 / 9' }}>
      {media}
    </div>
  )
}
