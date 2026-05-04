import Navbar        from '@/components/landing/Navbar'
import Hero          from '@/components/landing/Hero'
import HowItWorks    from '@/components/landing/HowItWorks'
import HeatmapPreview from '@/components/landing/HeatmapPreview'
import CtaSection    from '@/components/landing/CtaSection'
import Footer        from '@/components/landing/Footer'

export default function Home() {
  return (
    <main>
      <Navbar />
      <Hero />
      <HowItWorks />
      <HeatmapPreview />
      <CtaSection />
      <Footer />
    </main>
  )
}
