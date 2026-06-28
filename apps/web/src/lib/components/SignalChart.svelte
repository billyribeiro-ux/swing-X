<script lang="ts">
  import {
    CandlestickSeries,
    LineSeries,
    LineStyle,
    createChart,
    type IChartApi,
    type UTCTimestamp
  } from 'lightweight-charts';
  import type { Candle, LinePoint } from '$lib/fixtures/candles';
  import type { Side } from '@swing-x/shared-types';

  interface Props {
    candles: Candle[];
    vwap: LinePoint[];
    side: Side;
    entry: number;
    stop: number;
    target1: number;
    target2?: number;
  }

  let { candles, vwap, side, entry, stop, target1, target2 }: Props = $props();

  let container: HTMLDivElement;

  // Theme colors pulled from the app palette (oklch resolved to hex-ish equivalents
  // so the canvas renderer, which can't read CSS vars, matches the UI).
  const C = {
    up: '#2dd4a7',
    down: '#f4685f',
    grid: 'rgba(120,130,150,0.07)',
    text: '#8a93a6',
    bg: 'transparent',
    vwap: '#5aa9e6',
    target: '#2dd4a7',
    stop: '#f4685f',
    entry: '#c9d2e0'
  };

  $effect(() => {
    const chart: IChartApi = createChart(container, {
      layout: {
        background: { color: C.bg },
        textColor: C.text,
        fontFamily: "'JetBrains Mono', ui-monospace, monospace",
        fontSize: 11,
        attributionLogo: false
      },
      grid: {
        vertLines: { color: C.grid },
        horzLines: { color: C.grid }
      },
      rightPriceScale: { borderColor: 'rgba(120,130,150,0.15)' },
      timeScale: {
        borderColor: 'rgba(120,130,150,0.15)',
        timeVisible: true,
        secondsVisible: false
      },
      crosshair: { mode: 0 },
      autoSize: true
    });

    const candleSeries = chart.addSeries(CandlestickSeries, {
      upColor: C.up,
      downColor: C.down,
      borderUpColor: C.up,
      borderDownColor: C.down,
      wickUpColor: C.up,
      wickDownColor: C.down
    });
    candleSeries.setData(
      candles.map((c) => ({
        time: c.time as UTCTimestamp,
        open: c.open,
        high: c.high,
        low: c.low,
        close: c.close
      }))
    );

    const vwapSeries = chart.addSeries(LineSeries, {
      color: C.vwap,
      lineWidth: 1,
      lineStyle: LineStyle.Dotted,
      priceLineVisible: false,
      lastValueVisible: false,
      crosshairMarkerVisible: false
    });
    vwapSeries.setData(vwap.map((p) => ({ time: p.time as UTCTimestamp, value: p.value })));

    // Overlay level lines for entry / stop / targets.
    const line = (price: number, color: string, title: string, style: LineStyle) =>
      candleSeries.createPriceLine({
        price,
        color,
        lineWidth: 1,
        lineStyle: style,
        axisLabelVisible: true,
        title,
        lineVisible: true
      });

    line(entry, C.entry, `ENTRY ${side === 'long' ? '▲' : '▼'}`, LineStyle.Solid);
    line(stop, C.stop, 'STOP', LineStyle.Dashed);
    line(target1, C.target, 'T1', LineStyle.LargeDashed);
    if (target2 != null) {
      line(target2, C.target, 'T2', LineStyle.Dotted);
    }

    chart.timeScale().fitContent();

    return () => chart.remove();
  });
</script>

<div
  bind:this={container}
  class="h-[420px] w-full rounded-lg border border-base-800 bg-base-950/40"
  role="img"
  aria-label="Candlestick chart with VWAP overlay and entry, stop, and target level lines"
></div>
