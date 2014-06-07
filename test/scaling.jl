using Base.Test
import Images

@test Images.minclamp(-7, 10.0) === 10
@test Images.minclamp(-7, -10.0) === -7
@test Images.maxclamp(-7, 10.0) === -7
@test Images.maxclamp(-7, -10.0) === -10

@test Images.clamp(Uint8, -5) === 0x00
@test Images.clamp(Uint8, 5) === 0x05
@test Images.clamp(Uint8, 2000) === 0xff
@test Images.clamp(Float32, -3.2) === -3.2f0

@mytest_throws InexactError Images.clamp(Uint16, 3.2)
@test Images.truncround(Uint16, 3.2) === uint16(3)
@test Images.truncround(Uint16, -3.2) == uint16(0)
@test Images.truncround(Uint16, 1e20) == typemax(Uint16)
