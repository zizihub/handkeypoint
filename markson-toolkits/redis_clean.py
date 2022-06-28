import redis

face_embedding_redis_config = {
    "host": "r-bp14f0565e3316d4.redis.rds.aliyuncs.com",
    "port": 6379,
    "password": "ibmLUwbJ5o9E+Ve0x0hRaw==",
    "db": 1
}


face_url_redis_config = {
    "host": "r-bp1087acbac8ac04.redis.rds.aliyuncs.com",
    "port": 6379,
    "password": "ibmLUwbJ5o9E+Ve0x0hRaw==",
    "db": 1
}


def pool_factory(host, port, password, db):
    pool = redis.ConnectionPool(host=host,
                                port=port,
                                password=password,
                                db=int(db),
                                decode_responses=True)
    r = redis.Redis(connection_pool=pool)
    pipe = r.pipeline()
    return r, pipe


def main():
    face_embedding_redis, face_embedding_pipe = pool_factory(**face_embedding_redis_config)
    face_url_redis, face_url_pipe = pool_factory(**face_url_redis_config)

    print(face_embedding_redis)


if __name__ == "__main__":
    main()
