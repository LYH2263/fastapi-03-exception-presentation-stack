from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel

app = FastAPI()


class Item(BaseModel):
    name: str
    price: float | None = None
    owner_ids: list[int] | None = None


@app.get("/items/invalid", response_model=Item)
def get_invalid():
    return {"name": "invalid", "price": "foo"}


@app.get("/items/invalidnone", response_model=Item)
def get_invalid_none():
    return None


@app.get("/items/validnone", response_model=Item | None)
def get_valid_none(send_none: bool = False):
    if send_none:
        return None
    else:
        return {"name": "invalid", "price": 3.2}


@app.get("/items/innerinvalid", response_model=Item)
def get_innerinvalid():
    return {"name": "double invalid", "price": "foo", "owner_ids": ["foo", "bar"]}


@app.get("/items/invalidlist", response_model=list[Item])
def get_invalidlist():
    return [
        {"name": "foo"},
        {"name": "bar", "price": "bar"},
        {"name": "baz", "price": "baz"},
    ]


client = TestClient(app)


def test_invalid():
    # ResponseValidationError now has a stack-aware default presentation:
    # a 500 JSON response with the error list instead of an uncaught error.
    response = client.get("/items/invalid")
    assert response.status_code == 500
    assert response.headers["content-type"] == "application/json"
    assert response.json()["detail"][0]["loc"] == ["response", "price"]


def test_invalid_none():
    response = client.get("/items/invalidnone")
    assert response.status_code == 500
    assert response.json()["detail"][0]["loc"] == ["response"]


def test_valid_none_data():
    response = client.get("/items/validnone")
    data = response.json()
    assert response.status_code == 200
    assert data == {"name": "invalid", "price": 3.2, "owner_ids": None}


def test_valid_none_none():
    response = client.get("/items/validnone", params={"send_none": "true"})
    data = response.json()
    assert response.status_code == 200
    assert data is None


def test_double_invalid():
    response = client.get("/items/innerinvalid")
    assert response.status_code == 500
    locs = [error["loc"] for error in response.json()["detail"]]
    assert ["response", "price"] in locs
    assert ["response", "owner_ids", 0] in locs


def test_invalid_list():
    response = client.get("/items/invalidlist")
    assert response.status_code == 500
    locs = [error["loc"] for error in response.json()["detail"]]
    assert ["response", 1, "price"] in locs
    assert ["response", 2, "price"] in locs
