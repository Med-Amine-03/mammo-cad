'use client'

import { useState, useEffect } from 'react'
import { t, subscribeLang } from '@/lib/i18n'

export default function FooterBar() {
  const [, setLang] = useState('')
  useEffect(() => subscribeLang(setLang), [])

  return (
    <div className="footer-bar">
      <span style={{ fontWeight: 600, letterSpacing: '.04em' }}>MammoCAD</span>
      <span className="sep"/>
      <span style={{ color: 'var(--txt3)' }}>{t('footer.tagline')}</span>
      <span style={{ marginLeft: 'auto', color: 'var(--txt3)' }}>© 2026</span>
    </div>
  )
}
