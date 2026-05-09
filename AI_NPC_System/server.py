# server.py
import asyncio
import json
import time
import config
import fast_lane  # 기존 모듈
import slow_lane  # 기존 모듈

# 서버 설정
HOST = '127.0.0.1'
PORT = 5000

async def handle_client(reader, writer):
    addr = writer.get_extra_info('peername')
    print(f"[Server] 클라이언트 접속: {addr}")

    try:
        while True:
            # 1. Unity로부터 데이터 수신 (대기)
            data = await reader.read(4096)
            if not data:
                break
            
            user_text = data.decode('utf-8').strip()
            if not user_text:
                continue

            print(f"\n User Input: {user_text}")
            print("-" * 30)
            
            # ==================================================
            # [Fast Track] 감정 분석 & 키워드 추출 (CPU)
            # ==================================================
            start_time = time.time()
            
            fast_result = fast_lane.analyze_and_react(user_text)
            total_latency = time.time() - start_time
            
            tts_text = fast_result.get("tts_text") or fast_result["reaction"]

            fast_packet = {
                "schema_version": 2,
                "type": "fast",
                "emotion": fast_result['emotion_label'],
                "reaction": tts_text,
                "raw_reaction": fast_result['reaction'],
                "tts_text": tts_text,
                "echo_text": fast_result['echo_text'],
                "reaction_source": fast_result.get("reaction_source"),
                "bert_time": fast_result['bert_time'],
                "spacy_time": fast_result['spacy_time'],
                "strategy": fast_result.get('strategy'),
                "confidence_band": fast_result.get('confidence_band'),
                "top1": fast_result.get('top1'),
                "margin": fast_result.get('margin'),
                "entropy": fast_result.get('entropy'),
                "latency": f"{total_latency:.4f}s",
                "latency_ms": int(total_latency * 1000)
            }
            
            await send_json(writer, fast_packet)
            
            print(f"   [Fast Log]")
            print(f"   ├─ Time: {total_latency:.4f}s (BERT: {fast_result['bert_time']}, SpaCy: {fast_result['spacy_time']})")
            print(f"   ├─ Emotion: {fast_result['emotion_label']} ({fast_result.get('confidence_band')})")
            print(f"   ├─ Strategy: {fast_result.get('strategy')} / top1={fast_result.get('top1')} margin={fast_result.get('margin')} entropy={fast_result.get('entropy')}")
            print(f"   ├─ Reaction: \"{tts_text}\"")
            if fast_result['echo_text']:
                print(f"   └─ Echoing:  \"{fast_result['echo_text']}\"") # 서버 로그에도 표시
            
            # ==================================================
            #[Slow Track] LLM 심층 사고 (Network I/O)
            # ==================================================
            print("[Slow Lane] Gemini 2.5 Flash 생각 중...")
            
            # 4. Slow Lane 로직 수행 (비동기 대기)
            # Fast Lane의 결과(reaction)를 문맥으로 넘겨줍니다.
            llm_reply = await slow_lane.generate_response(
                user_text,
                tts_text,
                fast_result.get('strategy')
            )
            
            latency_slow = time.time() - start_time
            
            # 5. Slow Lane 패킷 생성
            slow_packet = {
                "schema_version": 2,
                "type": "slow",
                "npc_reply": llm_reply,
                "latency": f"{latency_slow:.4f}s",
                "latency_ms": int(latency_slow * 1000)
            }
            
            # 6. Unity로 발송
            await send_json(writer, slow_packet)
            print(f"[Slow Sent] {llm_reply} (Total: {latency_slow:.4f}s)")
            print("=" * 30)

    except Exception as e:
        print(f"Connection Error: {e}")
    finally:
        print(f"클라이언트 접속 종료: {addr}")
        writer.close()
        await writer.wait_closed()

async def send_json(writer, data_dict):
    """JSON 데이터를 보내고 즉시 버퍼를 비웁니다."""
    message = json.dumps(data_dict) + "\n" # 패킷 구분자
    writer.write(message.encode('utf-8'))
    await writer.drain() # 중요: 즉시 전송 보장

async def main():
    server = await asyncio.start_server(handle_client, HOST, PORT)
    print(f"[Pipeline Server] 가동 중... ({HOST}:{PORT})")
    print("   Unity 접속 대기 중...")
    async with server:
        await server.serve_forever()

if __name__ == "__main__":
    if hasattr(asyncio, 'WindowsSelectorEventLoopPolicy'):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
