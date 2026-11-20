/**
 * Internationalization (i18n) catalog for English ('en') and Sorani Kurdish ('ckb').
 * Provides RTL layout switching with LTR preservation for waveforms and numeric displays (D4.4).
 */

import { useSyncExternalStore } from 'react';

export type Locale = 'en' | 'ckb';

export interface TranslationCatalog {
  langToggleLabel: string;
  langToggleAria: string;
  noClip: string;
  uploading: string;
  clip: string;
  sent: string;
  analyzing: string;
  running: string;
  stage: string;
  unit: string;
  units: string;
  done: string;
  failed: string;
  reason: string;
  cancelled: string;
  armed: string;
  length: string;
  format: string;
  mono: string;
  stereo: string;
  channelsSuffix: string;
  noise: string;
  lufs: string;
  engineOffline: string;
  engineConnecting: string;
  engineReady: string;
  engineBusy: string;
  noAnalysis: string;
  profileStudioPro: string;
  profileDereverb: string;
  profileStudio: string;
  profileLowband: string;
  profileProduction: string;
  region: string;
  markStart: string;
  markEnd: string;
  clearRegion: string;

  // Header & shell
  modeSimple: string;
  modePro: string;
  modeSimpleTitle: string;
  modeProTitle: string;
  aiAssistant: string;
  aiAssistantTitle: string;
  drawerAiTitle: string;
  drawerBatchTitle: string;
  undo: string;
  redo: string;
  batchBtn: string;

  // Simple Studio Workspace
  welcomeTitle: string;
  welcomeSub: string;
  chooseFileBtn: string;
  welcomeFormats: string;
  micTestBtn: string;
  micRecordingPrefix: string;
  micStopEarly: string;
  micUnavailable: string;
  micDenied: string;
  processBtn: string;
  cancelBtn: string;
  play: string;
  pause: string;
  original: string;
  cleaned: string;
  holdForOriginal: string;
  holdOrigShort: string;
  holdForOriginalTitle: string;
  hearRemoved: string;
  hearRemovedTitle: string;
  saveCleanedAudio: string;
  saveCleanedVideo: string;
  saving: string;
  formatWav: string;
  formatAac: string;
  formatFlac: string;
  formatVideoMux: string;
  trustPrefix: string;
  trustLevelMatchOn: string;
  trustLevelMatchOff: string;
  stagePreflight: string;
  stageDecode: string;
  stageSegment: string;
  stageEnhance: string;
  stageGuard: string;
  stageFinish: string;
  stagePublish: string;
  stageDone: string;
  stageDefault: string;

  // Goal Presets
  goalsAriaLabel: string;
  goalAutoSmartTitle: string;
  goalAutoSmartDesc: string;
  goalVoiceMemoTitle: string;
  goalVoiceMemoDesc: string;
  goalPodcastTitle: string;
  goalPodcastDesc: string;
  goalRemoveEchoTitle: string;
  goalRemoveEchoDesc: string;
  goalFilmTitle: string;
  goalFilmDesc: string;
  presetAppliedPrefix: string;

  // Hero Macro Console
  quickControlsAria: string;
  naturalRoom: string;
  roomDry: string;
  roomSubtle: string;
  roomNatural: string;
  roomWarm: string;
  roomFull: string;
  roomHint: string;
  voicePolish: string;
  polishOff: string;
  polishHint: string;
  cleanAmount: string;
  cleanSub: string;
  autoPreviewOn: string;
  autoPreviewOff: string;
  autoPreviewTitle: string;
  livePreviewActive: string;
  audition8s: string;
  audition8sTitle: string;
  wetDryMix: string;
  realTimeBlend: string;
  activeAfterClean: string;
  deClick: string;
  deClip: string;

  // Audio Health Card
  healthListeningPrefix: string;
  healthListeningSuffix: string;
  healthNoiseReducedPrefix: string;
  healthNoiseReducedSuffix: string;
  healthVoiceCleaned: string;
  healthVoicePreserved: string;
  healthSafetyVerified: string;
  healthHeavyNoise: string;
  healthNoticeableNoise: string;
  healthLowNoise: string;
  healthVoiceQuiet: string;
  healthLoudRecording: string;
  healthDistortionPeaks: string;
  healthVideoWithAudio: string;
  healthAudioRecording: string;
  healthHeadlineDone: string;
  healthHeadlinePreview: string;
  healthHeadlineReady: string;

  // Pro Workspace & AudioStage
  proClean: string;
  proHpf: string;
  proDePlosive: string;
  proDeEss: string;
  proTonal: string;
  proLeveler: string;
  proUnits: string;
  proRuns: string;
  proBatch: string;
  proEnginePresets: string;
  proDeliveryMatch: string;
  proDeliveryPodcast: string;
  proDeliveryEbu: string;
  proDeliveryStream: string;
  proStereoAuto: string;
  proStereoMs: string;
  proStereoMono: string;
  proStereoSplit: string;
  drawerUnitsTitle: string;
  drawerRunsTitle: string;
  drawerEngineTitle: string;
  waveformLabel: string;
  spectrogramLabel: string;
  attenuate24: string;
  attenuate12: string;
  attenuating: string;
  askAi: string;
  clear: string;
}

export const CATALOGS: Record<Locale, TranslationCatalog> = {
  en: {
    langToggleLabel: 'کوردی',
    langToggleAria: 'کوردی — Switch to Sorani Kurdish',
    noClip: 'No clip',
    uploading: 'Uploading',
    clip: 'Clip',
    sent: 'Sent',
    analyzing: 'Analyzing',
    running: 'Running',
    stage: 'Stage',
    unit: 'Unit',
    units: 'Units',
    done: 'Done',
    failed: 'Failed',
    reason: 'Reason',
    cancelled: 'Cancelled',
    armed: 'Armed',
    length: 'Length',
    format: 'Format',
    mono: 'mono',
    stereo: 'stereo',
    channelsSuffix: 'ch',
    noise: 'Noise',
    lufs: 'LUFS',
    engineOffline: 'ENGINE OFFLINE',
    engineConnecting: 'ENGINE CONNECTING',
    engineReady: 'ENGINE READY',
    engineBusy: 'ENGINE BUSY',
    noAnalysis: 'No analysis',
    profileStudioPro: 'Studio Pro',
    profileDereverb: 'Dereverb',
    profileStudio: 'Studio',
    profileLowband: 'Low-band',
    profileProduction: 'Production',
    region: 'Region',
    markStart: 'In',
    markEnd: 'Out',
    clearRegion: 'Clear',

    modeSimple: 'Simple',
    modePro: 'Pro Studio',
    modeSimpleTitle: 'Simple Studio — effortless 3-step voice cleanup',
    modeProTitle: 'Pro Studio — spectrogram, spectrum HPF, surgical repair, and broadcast delivery',
    aiAssistant: 'AI Assistant',
    aiAssistantTitle: 'AI Audio Assistant (Cmd+K)',
    drawerAiTitle: 'AI Audio Assistant',
    drawerBatchTitle: 'Batch Queue',
    undo: 'Undo',
    redo: 'Redo',
    batchBtn: 'Batch…',

    welcomeTitle: 'Make any voice sound studio-recorded',
    welcomeSub:
      'Drop a phone voice memo, podcast, interview, or video here. HawaVoClean removes noise and room echo offline while protecting your natural voice.',
    chooseFileBtn: 'Choose Audio or Video',
    welcomeFormats: 'Supports WAV, M4A, MP3, FLAC, MP4, MOV · 100% private & offline',
    micTestBtn: 'Test with Mic (5s)',
    micRecordingPrefix: 'Recording…',
    micStopEarly: 'Click to stop early',
    micUnavailable: 'Microphone recording is not available in this browser.',
    micDenied: 'Microphone permission was denied — allow mic access to record a 5s test.',
    processBtn: 'Process',
    cancelBtn: 'Cancel',
    play: 'Play',
    pause: 'Pause',
    original: 'Original',
    cleaned: 'Cleaned',
    holdForOriginal: 'Hold for Original',
    holdOrigShort: 'Hold Orig',
    holdForOriginalTitle: 'Press and hold to briefly hear the raw original recording',
    hearRemoved: 'Hear Removed',
    hearRemovedTitle: 'Hear only the background noise & echo that was removed (D)',
    saveCleanedAudio: 'Save Cleaned Audio',
    saveCleanedVideo: 'Save Cleaned Video',
    saving: 'Saving…',
    formatWav: 'Best Quality (WAV)',
    formatAac: 'Small File (AAC / M4A)',
    formatFlac: 'Lossless (FLAC)',
    formatVideoMux: 'Cleaned Video (MP4)',
    trustPrefix: '100% Private & Offline · Your original file is never modified',
    trustLevelMatchOn: 'Level-matched comparison active',
    trustLevelMatchOff: 'Raw loudness comparison (level match off)',
    stagePreflight: 'Checking your recording…',
    stageDecode: 'Reading audio…',
    stageSegment: 'Finding speech sections…',
    stageEnhance: 'Removing background noise…',
    stageGuard: 'Verifying natural voice clarity…',
    stageFinish: 'Balancing loudness & polish…',
    stagePublish: 'Finalizing cleaned audio…',
    stageDone: 'Cleaning complete!',
    stageDefault: 'Cleaning your audio…',

    goalsAriaLabel: 'Cleaning goals',
    goalAutoSmartTitle: 'Auto Smart Clean',
    goalAutoSmartDesc: 'Picks the safest engine automatically',
    goalVoiceMemoTitle: 'Voice Memo Rescue',
    goalVoiceMemoDesc: 'Phone recordings & noisy rooms',
    goalPodcastTitle: 'Podcast & YouTube',
    goalPodcastDesc: 'Rich, consistent loudness (-16 LUFS)',
    goalRemoveEchoTitle: 'Remove Room Echo',
    goalRemoveEchoDesc: 'Tightens hollow rooms & reverb',
    goalFilmTitle: 'Film & Interview',
    goalFilmDesc: 'Natural ambience + broadcast target',
    presetAppliedPrefix: 'Preset applied:',

    quickControlsAria: 'Quick voice controls',
    naturalRoom: 'Natural Room',
    roomDry: 'Dry',
    roomSubtle: 'Subtle',
    roomNatural: 'Natural',
    roomWarm: 'Warm',
    roomFull: 'Full',
    roomHint: 'Keeps pauses natural',
    voicePolish: 'Voice Polish',
    polishOff: 'Off',
    polishHint: 'Smoothness & clarity',
    cleanAmount: 'Clean Amount',
    cleanSub: 'CLEAN',
    autoPreviewOn: 'Auto-Preview On',
    autoPreviewOff: 'Auto-Preview Off',
    autoPreviewTitle: 'Automatically audition an 8-second loop when you turn a dial',
    livePreviewActive: '● Live Preview Active',
    audition8s: 'Audition 8s (P)',
    audition8sTitle: 'Audition an 8-second preview right now (P)',
    wetDryMix: 'Wet / Dry Mix',
    realTimeBlend: 'Real-time blend',
    activeAfterClean: 'Active after clean',
    deClick: 'De-Click',
    deClip: 'De-Clip',

    healthListeningPrefix: 'Listening to “',
    healthListeningSuffix: '” to check background noise and clarity…',
    healthNoiseReducedPrefix: 'Background noise reduced (',
    healthNoiseReducedSuffix: ' dB quieter)',
    healthVoiceCleaned: 'Voice cleaned & balanced',
    healthVoicePreserved: 'Original voice preserved cleanly',
    healthSafetyVerified: 'Voice safety verified',
    healthHeavyNoise: 'Heavy background noise detected',
    healthNoticeableNoise: 'Noticeable background noise',
    healthLowNoise: 'Low background noise',
    healthVoiceQuiet: 'Voice is quiet — will gently boost',
    healthLoudRecording: 'Loud recording — headroom protected',
    healthDistortionPeaks: 'Some distortion peaks detected',
    healthVideoWithAudio: 'Video with audio',
    healthAudioRecording: 'Audio recording',
    healthHeadlineDone: 'Ready to compare — toggle Original / Cleaned below',
    healthHeadlinePreview: 'Auditioning live preview — click Process to apply to the full recording',
    healthHeadlineReady: 'Ready to clean — pick a goal or turn the Clean dial',

    proClean: 'Clean',
    proHpf: 'HPF',
    proDePlosive: 'De-Plosive',
    proDeEss: 'De-Ess',
    proTonal: 'Tonal',
    proLeveler: 'Leveler',
    proUnits: 'Units',
    proRuns: 'Runs',
    proBatch: 'Batch',
    proEnginePresets: 'Engine & Presets',
    proDeliveryMatch: 'Match',
    proDeliveryPodcast: 'Podcast -16',
    proDeliveryEbu: 'EBU -23',
    proDeliveryStream: 'Stream -14',
    proStereoAuto: 'Auto',
    proStereoMs: 'M/S',
    proStereoMono: 'Mono',
    proStereoSplit: 'Split',
    drawerUnitsTitle: 'Unit Guard Inspector',
    drawerRunsTitle: 'Session Runs',
    drawerEngineTitle: 'Engine, Restore & Presets',
    waveformLabel: 'Waveform',
    spectrogramLabel: 'Spectrogram',
    attenuate24: 'Attenuate −24 dB',
    attenuate12: '−12 dB',
    attenuating: 'Attenuating…',
    askAi: 'Ask AI',
    clear: 'Clear',
  },
  ckb: {
    langToggleLabel: 'English',
    langToggleAria: 'English — گۆڕین بۆ زمانی ئینگلیزی',
    noClip: 'هیچ کلیپێک نییە',
    uploading: 'بارکردن...',
    clip: 'کلیپ',
    sent: 'نێردراو',
    analyzing: 'شیکردنەوە...',
    running: 'کارپێکردن...',
    stage: 'قۆناغ',
    unit: 'یەکە',
    units: 'یەکەکان',
    done: 'تەواو',
    failed: 'شکستی هێنا',
    reason: 'هۆکار',
    cancelled: 'هەڵوەشێنرایەوە',
    armed: 'ئامادەکراو',
    length: 'درێژی',
    format: 'شێواز',
    mono: 'مۆنۆ',
    stereo: 'ستێریۆ',
    channelsSuffix: 'کەناڵ',
    noise: 'ژاوەژاو',
    lufs: 'LUFS',
    engineOffline: 'بزوێنەر لەسەر هێڵ نییە',
    engineConnecting: 'بزوێنەر پەیوەست دەبێت',
    engineReady: 'بزوێنەر ئامادەیە',
    engineBusy: 'بزوێنەر سەرقاڵە',
    noAnalysis: 'بێ شیکاری',
    profileStudioPro: 'ستۆدیۆ پرۆ',
    profileDereverb: 'لابردنی دەنگدانەوە',
    profileStudio: 'ستۆدیۆ',
    profileLowband: 'باندی نزم',
    profileProduction: 'بەرهەمهێنان',
    region: 'ناوچە',
    markStart: 'دەستپێک',
    markEnd: 'کۆتایی',
    clearRegion: 'سڕینەوە',

    modeSimple: 'سادە',
    modePro: 'ستۆدیۆی پرۆ',
    modeSimpleTitle: 'ستۆدیۆی سادە — پاککردنەوەی دەنگ بە ٣ هەنگاوی ئاسان',
    modeProTitle: 'ستۆدیۆی پرۆ — سپێکترۆگرام، فلتەری HPF، چاککردنەوەی ورد و پەخش',
    aiAssistant: 'یاریدەدەری ژیر',
    aiAssistantTitle: 'یاریدەدەری دەنگی ژیر (Cmd+K)',
    drawerAiTitle: 'یاریدەدەری دەنگی ژیر',
    drawerBatchTitle: 'ڕیزبەندی کۆمەڵە فایل',
    undo: 'گەڕانەوە',
    redo: 'دووبارەکردنەوە',
    batchBtn: 'کۆمەڵە…',

    welcomeTitle: 'هەر دەنگێک وەک ستۆدیۆ تۆمارکراو لێ بکە',
    welcomeSub:
      'تۆمارێکی مۆبایل، پۆدکاست، چاوپێکەوتن، یان ڤیدیۆیەک بخەرە ئێرە. هاواڤۆکلین ژاوەژاو و دەنگدانەوەی ژوور بەبێ ئینتەرنێت لادەبات و سروشتیبوونی دەنگت دەپارێزێت.',
    chooseFileBtn: 'دەنگ یان ڤیدیۆ هەڵبژێرە',
    welcomeFormats: 'پشتگیری WAV، M4A، MP3، FLAC، MP4، MOV · ١٠٠٪ تایبەت و بێ ئینتەرنێت',
    micTestBtn: 'تاقیکردنەوە بە مایک (٥ چرکە)',
    micRecordingPrefix: 'تۆمارکردن…',
    micStopEarly: 'کرتە بکە بۆ وەستاندنی زوو',
    micUnavailable: 'تۆمارکردن بە مایکرۆفۆن لەم وێبگەڕەدا بەردەست نییە.',
    micDenied: 'ڕێگەپێدانی مایکرۆفۆن ڕەتکرایەوە — ڕێگە بە مایک بدە بۆ تاقیکردنەوەی ٥ چرکەیی.',
    processBtn: 'پاککردنەوە',
    cancelBtn: 'هەڵوەشاندنەوە',
    play: 'لێدان',
    pause: 'وەستان',
    original: 'سەرەکی',
    cleaned: 'پاککراوە',
    holdForOriginal: 'ڕایگرە بۆ سەرەکی',
    holdOrigShort: 'سەرەکی بگرە',
    holdForOriginalTitle: 'دایگرە و ڕایگرە بۆ بیستنی دەنگە سەرەکییەکە',
    hearRemoved: 'بیستنی لابراوەکان',
    hearRemovedTitle: 'تەنها ئەو ژاوەژاو و دەنگدانەوەیە ببیستە کە لابراوە (D)',
    saveCleanedAudio: 'پاشەکەوتکردنی دەنگی پاککراوە',
    saveCleanedVideo: 'پاشەکەوتکردنی ڤیدیۆی پاککراوە',
    saving: 'پاشەکەوتکردن…',
    formatWav: 'باشترین کوالێتی (WAV)',
    formatAac: 'فایلی بچووک (AAC / M4A)',
    formatFlac: 'بێ کەمبوونەوە (FLAC)',
    formatVideoMux: 'ڤیدیۆی پاککراوە (MP4)',
    trustPrefix: '١٠٠٪ تایبەت و بێ ئینتەرنێت · فایلە سەرەکییەکەت هەرگیز ناگۆڕدرێت',
    trustLevelMatchOn: 'هاوسەنگکردنی بەرزی دەنگ چالاکە',
    trustLevelMatchOff: 'بەراوردکردنی بەرزی خاو (هاوسەنگکردن ناچالاکە)',
    stagePreflight: 'پشکنینی تۆمارەکەت…',
    stageDecode: 'خوێندنەوەی دەنگ…',
    stageSegment: 'دۆزینەوەی بەشەکانی قسەکردن…',
    stageEnhance: 'لابردنی ژاوەژاوی باکگراوند…',
    stageGuard: 'دڵنیابوونەوە لە ڕوونی سروشتی دەنگ…',
    stageFinish: 'هاوسەنگکردنی بەرزی و ڕوونی…',
    stagePublish: 'ئامادەکردنی کۆتایی دەنگی پاککراوە…',
    stageDone: 'پاککردنەوە تەواو بوو!',
    stageDefault: 'پاککردنەوەی دەنگەکەت…',

    goalsAriaLabel: 'ئامانجەکانی پاککردنەوە',
    goalAutoSmartTitle: 'پاککردنەوەی زیرەکی خۆکار',
    goalAutoSmartDesc: 'سەلامەتترین بزوێنەر بە خۆکاری هەڵدەبژێرێت',
    goalVoiceMemoTitle: 'چاککردنەوەی تۆماری مۆبایل',
    goalVoiceMemoDesc: 'تۆماری مۆبایل و ژووری پڕ ژاوەژاو',
    goalPodcastTitle: 'پۆدکاست و یوتیوب',
    goalPodcastDesc: 'دەنگی دەوڵەمەند و هاوسەنگ (-16 LUFS)',
    goalRemoveEchoTitle: 'لابردنی دەنگدانەوەی ژوور',
    goalRemoveEchoDesc: 'ژووری بۆش و دەنگدانەوە کەم دەکاتەوە',
    goalFilmTitle: 'فیلم و چاوپێکەوتن',
    goalFilmDesc: 'کەشی سروشتی + ستانداردی پەخش',
    presetAppliedPrefix: 'ڕێکخستن جێبەجێ کرا:',

    quickControlsAria: 'کۆنتڕۆڵە خێراکانی دەنگ',
    naturalRoom: 'کەشی ژوور',
    roomDry: 'وشک',
    roomSubtle: 'کەم',
    roomNatural: 'سروشتی',
    roomWarm: 'گەرم',
    roomFull: 'پڕ',
    roomHint: 'وەستانەکان سروشتی دەهێڵێتەوە',
    voicePolish: 'ڕوونکردنەوەی دەنگ',
    polishOff: 'ناچالاک',
    polishHint: 'نەرمی و ڕوونی',
    cleanAmount: 'ڕێژەی پاککردنەوە',
    cleanSub: 'پاککردن',
    autoPreviewOn: 'تاقیکردنەوەی خۆکار: چالاک',
    autoPreviewOff: 'تاقیکردنەوەی خۆکار: ناچالاک',
    autoPreviewTitle: 'بە خۆکاری ٨ چرکە گوێ بگرە کاتێک کۆنتڕۆڵێک دەگۆڕیت',
    livePreviewActive: '● پێشبینینی ڕاستەوخۆ چالاکە',
    audition8s: 'گوێگرتنی ٨ چرکە (P)',
    audition8sTitle: 'ئێستا گوێ لە پێشبینینێکی ٨ چرکەیی بگرە (P)',
    wetDryMix: 'تێکەڵکردنی پاک/سەرەکی',
    realTimeBlend: 'تێکەڵکردنی ڕاستەوخۆ',
    activeAfterClean: 'دوای پاککردنەوە چالاک دەبێت',
    deClick: 'لابردنی کرتە',
    deClip: 'چاککردنی تەقین',

    healthListeningPrefix: 'گوێگرتن لە «',
    healthListeningSuffix: '» بۆ پشکنینی ژاوەژاو و ڕوونی…',
    healthNoiseReducedPrefix: 'ژاوەژاوی باکگراوند کەمکرایەوە (',
    healthNoiseReducedSuffix: ' dB بێدەنگتر)',
    healthVoiceCleaned: 'دەنگ پاککرایەوە و هاوسەنگ کرا',
    healthVoicePreserved: 'دەنگە سەرەکییەکە بە پاکی پارێزرا',
    healthSafetyVerified: 'سەلامەتی دەنگ پشتڕاست کرایەوە',
    healthHeavyNoise: 'ژاوەژاوی زۆری باکگراوند دۆزرایەوە',
    healthNoticeableNoise: 'ژاوەژاوی دیاری باکگراوند',
    healthLowNoise: 'ژاوەژاوی باکگراوند کەمە',
    healthVoiceQuiet: 'دەنگ نزمە — بە نەرمی بەرز دەکرێتەوە',
    healthLoudRecording: 'تۆماری بەرز — پارێزگاری لە لوتکە کرا',
    healthDistortionPeaks: 'هەندێک لوتکەی شێواوی دۆزرایەوە',
    healthVideoWithAudio: 'ڤیدیۆ بە دەنگەوە',
    healthAudioRecording: 'تۆماری دەنگی',
    healthHeadlineDone: 'ئامادەیە بۆ بەراورد — لە خوارەوە لە نێوان سەرەکی و پاککراوەدا بگۆڕە',
    healthHeadlinePreview: 'گوێگرتن لە پێشبینین — کرتە لە پاککردنەوە بکە بۆ جێبەجێکردن لەسەر هەموو تۆمارەکە',
    healthHeadlineReady: 'ئامادەیە بۆ پاککردنەوە — ئامانجێک هەڵبژێرە یان کۆنتڕۆڵی پاککردن بسوڕێنە',

    proClean: 'پاککردن',
    proHpf: 'HPF',
    proDePlosive: 'لابردنی تەقە',
    proDeEss: 'کەمکردنی س',
    proTonal: 'تۆن',
    proLeveler: 'هاوسەنگکەر',
    proUnits: 'یەکەکان',
    proRuns: 'کارەکان',
    proBatch: 'کۆمەڵە',
    proEnginePresets: 'بزوێنەر و پێشڕێکخستن',
    proDeliveryMatch: 'هاوشێوە',
    proDeliveryPodcast: 'پۆدکاست -16',
    proDeliveryEbu: 'EBU -23',
    proDeliveryStream: 'پەخش -14',
    proStereoAuto: 'خۆکار',
    proStereoMs: 'M/S',
    proStereoMono: 'مۆنۆ',
    proStereoSplit: 'جیاکردنەوە',
    drawerUnitsTitle: 'پشکنەری پاراستنی یەکەکان',
    drawerRunsTitle: 'کارەکانی ئەم دانیشتنە',
    drawerEngineTitle: 'بزوێنەر، گێڕانەوە و پێشڕێکخستن',
    waveformLabel: 'شەپۆلی دەنگ',
    spectrogramLabel: 'سپێکترۆگرام',
    attenuate24: 'کەمکردنەوە −24 dB',
    attenuate12: '−12 dB',
    attenuating: 'کەمکردنەوە…',
    askAi: 'پرسیار لە ژیر',
    clear: 'سڕینەوە',
  },
};

const STORAGE_KEY = 'hawavoclean_locale';

function getInitialLocale(): Locale {
  if (typeof window !== 'undefined' && window.localStorage) {
    try {
      const stored = window.localStorage.getItem(STORAGE_KEY);
      if (stored === 'en' || stored === 'ckb') {
        return stored;
      }
    } catch {
      // Ignore storage access restrictions
    }
  }
  return 'en';
}

function applyDocumentLocale(locale: Locale): void {
  if (typeof document !== 'undefined' && document.documentElement) {
    document.documentElement.lang = locale;
    document.documentElement.dir = locale === 'ckb' ? 'rtl' : 'ltr';
  }
}

let currentLocale: Locale = getInitialLocale();
applyDocumentLocale(currentLocale);

const listeners = new Set<() => void>();

/** Re-read persisted locale from localStorage and apply dir/lang immediately. */
export function initLocaleFromStorage(): Locale {
  currentLocale = getInitialLocale();
  applyDocumentLocale(currentLocale);
  for (const listener of listeners) {
    listener();
  }
  return currentLocale;
}

export function setLocale(locale: Locale): void {
  currentLocale = locale;
  if (typeof window !== 'undefined' && window.localStorage) {
    try {
      window.localStorage.setItem(STORAGE_KEY, locale);
    } catch {
      // Ignore storage access restrictions
    }
  }
  applyDocumentLocale(locale);
  for (const listener of listeners) {
    listener();
  }
}

export function getLocale(): Locale {
  return currentLocale;
}

export function useI18n(): {
  locale: Locale;
  t: TranslationCatalog;
  setLocale: (locale: Locale) => void;
  isRtl: boolean;
} {
  const locale = useSyncExternalStore(
    (cb) => {
      listeners.add(cb);
      return () => listeners.delete(cb);
    },
    () => currentLocale,
    () => 'en' as Locale,
  );

  return {
    locale,
    t: CATALOGS[locale] ?? CATALOGS.en,
    setLocale,
    isRtl: locale === 'ckb',
  };
}
