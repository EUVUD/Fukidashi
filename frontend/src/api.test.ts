import { describe, expect, it } from 'vitest'
import { EVENT_KINDS, parseEvent } from './api'

// What the browser hands over for one event of the stream: its `event:` name, `lastEventId` (the `id:` field, the event's
// seq) and `data`. The importing of `api.ts` needs no DOM: `EventSource` is only touched inside `watchRun`.
describe('parseEvent', () => {
  it('reads the seq from the last event id and the data as JSON', () => {
    expect(parseEvent('stage_start', '12', '{"stage":"review","index":1}')).toEqual({
      seq: 12,
      kind: 'stage_start',
      data: { stage: 'review', index: 1 },
    })
  })

  it('keeps Chinese and Japanese text as it is', () => {
    const event = parseEvent('tool_result', '3', '{"name":"lookup_term","result":"コップ → 杯子","error":null}')
    expect(event?.kind === 'tool_result' && event.data.result).toBe('コップ → 杯子')
  })

  it('drops an event whose id is not a positive whole number', () => {
    for (const id of ['', '0', '-1', '1.5', '1e3', ' 4', '4 ', '0x3', 'abc', '99999999999999999999']) {
      expect(parseEvent('model_call', id, '{}'), JSON.stringify(id)).toBeNull()
    }
    expect(parseEvent('model_call', '007', '{}')?.seq).toBe(7)
  })

  it('drops an event whose data is not a JSON object', () => {
    for (const data of ['', 'not json', 'null', '7', '"text"', '[1, 2]', '{"cut":']) {
      expect(parseEvent('model_call', '1', data), data).toBeNull()
    }
  })

  it('has a kind for each of the eight events of the contract', () => {
    expect([...EVENT_KINDS].sort()).toEqual(
      ['fallback', 'model_call', 'repair', 'run_end', 'run_start', 'stage_end', 'stage_start', 'tool_result'],
    )
  })
})
