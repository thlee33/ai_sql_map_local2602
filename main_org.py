import os
import json
import re
import duckdb
import ollama
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pyproj import Transformer

app = FastAPI()

# 1. 경로 및 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data").replace("\\", "/")
HTML_PATH = os.path.join(BASE_DIR, "index.html")
MODEL_NAME = "gpt-oss:20b" # qwen2.5:7b"  

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class QueryRequest(BaseModel):
    text: str

def get_db():
    conn = duckdb.connect()
    conn.execute("INSTALL spatial; LOAD spatial;")
    return conn

def extract_mart_name(text: str) -> str:
    """AI 응답 파싱 개선 - JSON 파싱 실패 시 정규식으로 폴백"""
    system_prompt = """
너는 데이터 추출 전문가야. 사용자 질문에서 '마트 이름과 지점명'을 모두 추출해.
반드시 이 형식으로만 답해: {"mart_name": "마트이름 지점명"}

예시:
- 입력: "롯데마트 서울역점에서 가장 가까운 소방서" → 출력: {"mart_name": "롯데마트 서울역"}
- 입력: "이마트 용산점 근처 소방서" → 출력: {"mart_name": "이마트 용산"}
- 입력: "GS25 명동점" → 출력: {"mart_name": "GS25 명동"}

중요: "점"은 제외하고 지점명까지 포함해서 추출해.
"""
    
    try:
        response = ollama.chat(
            model=MODEL_NAME, 
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': text}
            ], 
            options={'temperature': 0}
        )
        
        content = response['message']['content']
        print(f"[AI Response] {content}")
        
        # JSON 파싱 시도
        try:
            plan = json.loads(content)
            return plan.get('mart_name', '')
        except:
            # JSON 파싱 실패 시 정규식으로 추출
            match = re.search(r'"mart_name"\s*:\s*"([^"]+)"', content)
            if match:
                return match.group(1)
            
            # 정규식으로 원본 텍스트에서 "마트명 + 지점명" 추출
            # 예: "롯데마트 서울역점" → "롯데마트 서울역"
            mart_pattern = r'(롯데마트|이마트|GS25|CU|세븐일레븐|홈플러스)\s*([가-힣]+)점?'
            match = re.search(mart_pattern, text)
            if match:
                return f"{match.group(1)} {match.group(2)}"
            
            # 마트명만 추출
            for kw in ['롯데마트', '이마트', 'GS25', 'CU', '세븐일레븐']:
                if kw in text:
                    return kw
            return text.split()[0] if text.split() else ''
    except Exception as e:
        print(f"[AI Error] {e}")
        # AI 완전 실패 시 정규식으로 직접 추출
        mart_pattern = r'(롯데마트|이마트|GS25|CU|세븐일레븐|홈플러스)\s*([가-힣]+)점?'
        match = re.search(mart_pattern, text)
        if match:
            return f"{match.group(1)} {match.group(2)}"
        
        for kw in ['롯데마트', '이마트', 'GS', 'CU']:
            if kw in text:
                return kw
        return ''

@app.get("/")
async def read_index():
    return FileResponse(HTML_PATH)

@app.post("/analyze")
async def analyze_query(request: QueryRequest):
    conn = get_db()
    
    try:
        # 1. AI로 마트명 추출
        search_term = extract_mart_name(request.text)
        print(f"[검색어] {search_term}")
        
        if not search_term:
            return {"answer_text": "마트 이름을 인식할 수 없습니다."}

        # 2. 마트 위치 검색 (파라미터 바인딩으로 SQL 인젝션 방지)
        mart_sql = """
            SELECT nam, ST_AsText(geometry) as wkt, geometry 
            FROM read_parquet(?) 
            WHERE nam ILIKE ? 
            LIMIT 1
        """
        mart_data = conn.execute(
            mart_sql, 
            [f"{DATA_DIR}/mart.parquet", f"%{search_term}%"]
        ).fetchone()

        if not mart_data:
            return {"answer_text": f"'{search_term}'을(를) 찾을 수 없습니다. 데이터를 확인해주세요."}

        mart_name, mart_wkt, mart_geom = mart_data
        print(f"[마트 발견] {mart_name} at {mart_wkt}")

        # 3. 가장 가까운 소방서 검색
        fire_sql = """
            SELECT nam, ST_AsText(geometry) as wkt, geometry,
                   ST_Distance(geometry, ST_GeomFromText(?)) as distance
            FROM read_parquet(?)
            ORDER BY distance ASC 
            LIMIT 1
        """
        fire_data = conn.execute(
            fire_sql,
            [mart_wkt, f"{DATA_DIR}/firestation.parquet"]
        ).fetchone()

        if not fire_data:
            return {"answer_text": "소방서 데이터를 찾을 수 없습니다."}

        fire_name, fire_wkt = fire_data[0], fire_data[1]
        distance = fire_data[3]
        print(f"[소방서 발견] {fire_name} (거리: {distance:.2f}m)")

        # 4. GeoJSON 생성 (EPSG:5179 → EPSG:4326 변환)
        # 방법 1: 파이썬에서 직접 변환 후 WKT 생성
        from pyproj import Transformer
        
        # EPSG:5179 -> EPSG:4326 변환기
        transformer = Transformer.from_crs("EPSG:5179", "EPSG:4326", always_xy=True)
        
        # 마트 좌표 변환
        mart_coords = mart_geom  # geometry 객체 직접 사용
        mart_x = conn.execute(f"SELECT ST_X(ST_GeomFromText('{mart_wkt}'))").fetchone()[0]
        mart_y = conn.execute(f"SELECT ST_Y(ST_GeomFromText('{mart_wkt}'))").fetchone()[0]
        mart_lon, mart_lat = transformer.transform(mart_x, mart_y)
        
        # 소방서 좌표 변환
        fire_x = conn.execute(f"SELECT ST_X(ST_GeomFromText('{fire_wkt}'))").fetchone()[0]
        fire_y = conn.execute(f"SELECT ST_Y(ST_GeomFromText('{fire_wkt}'))").fetchone()[0]
        fire_lon, fire_lat = transformer.transform(fire_x, fire_y)
        
        print(f"[좌표 변환] 마트: ({mart_lon:.6f}, {mart_lat:.6f}), 소방서: ({fire_lon:.6f}, {fire_lat:.6f})")
        
        # WGS84 좌표로 GeoJSON 생성
        geojson_sql = """
            SELECT json_object(
                'type', 'FeatureCollection',
                'features', json_group_array(json_object(
                    'type', 'Feature',
                    'geometry', ST_AsGeoJSON(geometry)::JSON,
                    'properties', json_object('display_name', obj_name, 'type', type)
                ))
            )
            FROM (
                SELECT ? as obj_name, 'mart' as type, ST_GeomFromText(?) as geometry
                UNION ALL
                SELECT ? as obj_name, 'firestation' as type, ST_GeomFromText(?) as geometry
            ) t
        """
        
        mart_wkt_4326 = f"POINT ({mart_lon} {mart_lat})"
        fire_wkt_4326 = f"POINT ({fire_lon} {fire_lat})"
        
        geojson_res = conn.execute(
            geojson_sql, 
            [mart_name, mart_wkt_4326, fire_name, fire_wkt_4326]
        ).fetchone()[0]
        
        result = json.loads(geojson_res)
        result['summary'] = f"{mart_name}에서 가장 가까운 소방서는 {fire_name}입니다 (거리: {distance:.0f}m)"
        
        return result

    except Exception as e:
        print(f"[Error] {e}")
        import traceback
        traceback.print_exc()
        return {"answer_text": f"오류 발생: {str(e)}"}
    finally:
        conn.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
