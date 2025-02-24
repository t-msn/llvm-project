; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc -mtriple=riscv32 -mattr=+d,+experimental-zfh,+experimental-v -target-abi=ilp32d \
; RUN:     -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=riscv64 -mattr=+d,+experimental-zfh,+experimental-v -target-abi=lp64d \
; RUN:     -verify-machineinstrs < %s | FileCheck %s

declare half @llvm.vp.reduce.fadd.nxv1f16(half, <vscale x 1 x half>, <vscale x 1 x i1>, i32)

define half @vpreduce_fadd_nxv1f16(half %s, <vscale x 1 x half> %v, <vscale x 1 x i1> %m, i32 zeroext %evl) {
; CHECK-LABEL: vpreduce_fadd_nxv1f16:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetvli a1, zero, e16, m1, ta, mu
; CHECK-NEXT:    vfmv.v.f v25, fa0
; CHECK-NEXT:    vsetvli zero, a0, e16, mf4, tu, mu
; CHECK-NEXT:    vfredsum.vs v25, v8, v25, v0.t
; CHECK-NEXT:    vfmv.f.s fa0, v25
; CHECK-NEXT:    ret
  %r = call reassoc half @llvm.vp.reduce.fadd.nxv1f16(half %s, <vscale x 1 x half> %v, <vscale x 1 x i1> %m, i32 %evl)
  ret half %r
}

define half @vpreduce_ord_fadd_nxv1f16(half %s, <vscale x 1 x half> %v, <vscale x 1 x i1> %m, i32 zeroext %evl) {
; CHECK-LABEL: vpreduce_ord_fadd_nxv1f16:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetvli a1, zero, e16, m1, ta, mu
; CHECK-NEXT:    vfmv.v.f v25, fa0
; CHECK-NEXT:    vsetvli zero, a0, e16, mf4, tu, mu
; CHECK-NEXT:    vfredosum.vs v25, v8, v25, v0.t
; CHECK-NEXT:    vfmv.f.s fa0, v25
; CHECK-NEXT:    ret
  %r = call half @llvm.vp.reduce.fadd.nxv1f16(half %s, <vscale x 1 x half> %v, <vscale x 1 x i1> %m, i32 %evl)
  ret half %r
}

declare half @llvm.vp.reduce.fadd.nxv2f16(half, <vscale x 2 x half>, <vscale x 2 x i1>, i32)

define half @vpreduce_fadd_nxv2f16(half %s, <vscale x 2 x half> %v, <vscale x 2 x i1> %m, i32 zeroext %evl) {
; CHECK-LABEL: vpreduce_fadd_nxv2f16:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetvli a1, zero, e16, m1, ta, mu
; CHECK-NEXT:    vfmv.v.f v25, fa0
; CHECK-NEXT:    vsetvli zero, a0, e16, mf2, tu, mu
; CHECK-NEXT:    vfredsum.vs v25, v8, v25, v0.t
; CHECK-NEXT:    vfmv.f.s fa0, v25
; CHECK-NEXT:    ret
  %r = call reassoc half @llvm.vp.reduce.fadd.nxv2f16(half %s, <vscale x 2 x half> %v, <vscale x 2 x i1> %m, i32 %evl)
  ret half %r
}

define half @vpreduce_ord_fadd_nxv2f16(half %s, <vscale x 2 x half> %v, <vscale x 2 x i1> %m, i32 zeroext %evl) {
; CHECK-LABEL: vpreduce_ord_fadd_nxv2f16:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetvli a1, zero, e16, m1, ta, mu
; CHECK-NEXT:    vfmv.v.f v25, fa0
; CHECK-NEXT:    vsetvli zero, a0, e16, mf2, tu, mu
; CHECK-NEXT:    vfredosum.vs v25, v8, v25, v0.t
; CHECK-NEXT:    vfmv.f.s fa0, v25
; CHECK-NEXT:    ret
  %r = call half @llvm.vp.reduce.fadd.nxv2f16(half %s, <vscale x 2 x half> %v, <vscale x 2 x i1> %m, i32 %evl)
  ret half %r
}

declare half @llvm.vp.reduce.fadd.nxv4f16(half, <vscale x 4 x half>, <vscale x 4 x i1>, i32)

define half @vpreduce_fadd_nxv4f16(half %s, <vscale x 4 x half> %v, <vscale x 4 x i1> %m, i32 zeroext %evl) {
; CHECK-LABEL: vpreduce_fadd_nxv4f16:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetvli a1, zero, e16, m1, ta, mu
; CHECK-NEXT:    vfmv.v.f v25, fa0
; CHECK-NEXT:    vsetvli zero, a0, e16, m1, tu, mu
; CHECK-NEXT:    vfredsum.vs v25, v8, v25, v0.t
; CHECK-NEXT:    vfmv.f.s fa0, v25
; CHECK-NEXT:    ret
  %r = call reassoc half @llvm.vp.reduce.fadd.nxv4f16(half %s, <vscale x 4 x half> %v, <vscale x 4 x i1> %m, i32 %evl)
  ret half %r
}

define half @vpreduce_ord_fadd_nxv4f16(half %s, <vscale x 4 x half> %v, <vscale x 4 x i1> %m, i32 zeroext %evl) {
; CHECK-LABEL: vpreduce_ord_fadd_nxv4f16:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetvli a1, zero, e16, m1, ta, mu
; CHECK-NEXT:    vfmv.v.f v25, fa0
; CHECK-NEXT:    vsetvli zero, a0, e16, m1, tu, mu
; CHECK-NEXT:    vfredosum.vs v25, v8, v25, v0.t
; CHECK-NEXT:    vfmv.f.s fa0, v25
; CHECK-NEXT:    ret
  %r = call half @llvm.vp.reduce.fadd.nxv4f16(half %s, <vscale x 4 x half> %v, <vscale x 4 x i1> %m, i32 %evl)
  ret half %r
}

declare float @llvm.vp.reduce.fadd.nxv1f32(float, <vscale x 1 x float>, <vscale x 1 x i1>, i32)

define float @vpreduce_fadd_nxv1f32(float %s, <vscale x 1 x float> %v, <vscale x 1 x i1> %m, i32 zeroext %evl) {
; CHECK-LABEL: vpreduce_fadd_nxv1f32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetvli a1, zero, e32, m1, ta, mu
; CHECK-NEXT:    vfmv.v.f v25, fa0
; CHECK-NEXT:    vsetvli zero, a0, e32, mf2, tu, mu
; CHECK-NEXT:    vfredsum.vs v25, v8, v25, v0.t
; CHECK-NEXT:    vfmv.f.s fa0, v25
; CHECK-NEXT:    ret
  %r = call reassoc float @llvm.vp.reduce.fadd.nxv1f32(float %s, <vscale x 1 x float> %v, <vscale x 1 x i1> %m, i32 %evl)
  ret float %r
}

define float @vpreduce_ord_fadd_nxv1f32(float %s, <vscale x 1 x float> %v, <vscale x 1 x i1> %m, i32 zeroext %evl) {
; CHECK-LABEL: vpreduce_ord_fadd_nxv1f32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetvli a1, zero, e32, m1, ta, mu
; CHECK-NEXT:    vfmv.v.f v25, fa0
; CHECK-NEXT:    vsetvli zero, a0, e32, mf2, tu, mu
; CHECK-NEXT:    vfredosum.vs v25, v8, v25, v0.t
; CHECK-NEXT:    vfmv.f.s fa0, v25
; CHECK-NEXT:    ret
  %r = call float @llvm.vp.reduce.fadd.nxv1f32(float %s, <vscale x 1 x float> %v, <vscale x 1 x i1> %m, i32 %evl)
  ret float %r
}

declare float @llvm.vp.reduce.fadd.nxv2f32(float, <vscale x 2 x float>, <vscale x 2 x i1>, i32)

define float @vpreduce_fadd_nxv2f32(float %s, <vscale x 2 x float> %v, <vscale x 2 x i1> %m, i32 zeroext %evl) {
; CHECK-LABEL: vpreduce_fadd_nxv2f32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetvli a1, zero, e32, m1, ta, mu
; CHECK-NEXT:    vfmv.v.f v25, fa0
; CHECK-NEXT:    vsetvli zero, a0, e32, m1, tu, mu
; CHECK-NEXT:    vfredsum.vs v25, v8, v25, v0.t
; CHECK-NEXT:    vfmv.f.s fa0, v25
; CHECK-NEXT:    ret
  %r = call reassoc float @llvm.vp.reduce.fadd.nxv2f32(float %s, <vscale x 2 x float> %v, <vscale x 2 x i1> %m, i32 %evl)
  ret float %r
}

define float @vpreduce_ord_fadd_nxv2f32(float %s, <vscale x 2 x float> %v, <vscale x 2 x i1> %m, i32 zeroext %evl) {
; CHECK-LABEL: vpreduce_ord_fadd_nxv2f32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetvli a1, zero, e32, m1, ta, mu
; CHECK-NEXT:    vfmv.v.f v25, fa0
; CHECK-NEXT:    vsetvli zero, a0, e32, m1, tu, mu
; CHECK-NEXT:    vfredosum.vs v25, v8, v25, v0.t
; CHECK-NEXT:    vfmv.f.s fa0, v25
; CHECK-NEXT:    ret
  %r = call float @llvm.vp.reduce.fadd.nxv2f32(float %s, <vscale x 2 x float> %v, <vscale x 2 x i1> %m, i32 %evl)
  ret float %r
}

declare float @llvm.vp.reduce.fadd.nxv4f32(float, <vscale x 4 x float>, <vscale x 4 x i1>, i32)

define float @vpreduce_fadd_nxv4f32(float %s, <vscale x 4 x float> %v, <vscale x 4 x i1> %m, i32 zeroext %evl) {
; CHECK-LABEL: vpreduce_fadd_nxv4f32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetvli a1, zero, e32, m1, ta, mu
; CHECK-NEXT:    vfmv.v.f v25, fa0
; CHECK-NEXT:    vsetvli zero, a0, e32, m2, tu, mu
; CHECK-NEXT:    vfredsum.vs v25, v8, v25, v0.t
; CHECK-NEXT:    vfmv.f.s fa0, v25
; CHECK-NEXT:    ret
  %r = call reassoc float @llvm.vp.reduce.fadd.nxv4f32(float %s, <vscale x 4 x float> %v, <vscale x 4 x i1> %m, i32 %evl)
  ret float %r
}

define float @vpreduce_ord_fadd_nxv4f32(float %s, <vscale x 4 x float> %v, <vscale x 4 x i1> %m, i32 zeroext %evl) {
; CHECK-LABEL: vpreduce_ord_fadd_nxv4f32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetvli a1, zero, e32, m1, ta, mu
; CHECK-NEXT:    vfmv.v.f v25, fa0
; CHECK-NEXT:    vsetvli zero, a0, e32, m2, tu, mu
; CHECK-NEXT:    vfredosum.vs v25, v8, v25, v0.t
; CHECK-NEXT:    vfmv.f.s fa0, v25
; CHECK-NEXT:    ret
  %r = call float @llvm.vp.reduce.fadd.nxv4f32(float %s, <vscale x 4 x float> %v, <vscale x 4 x i1> %m, i32 %evl)
  ret float %r
}

declare double @llvm.vp.reduce.fadd.nxv1f64(double, <vscale x 1 x double>, <vscale x 1 x i1>, i32)

define double @vpreduce_fadd_nxv1f64(double %s, <vscale x 1 x double> %v, <vscale x 1 x i1> %m, i32 zeroext %evl) {
; CHECK-LABEL: vpreduce_fadd_nxv1f64:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetvli a1, zero, e64, m1, ta, mu
; CHECK-NEXT:    vfmv.v.f v25, fa0
; CHECK-NEXT:    vsetvli zero, a0, e64, m1, tu, mu
; CHECK-NEXT:    vfredsum.vs v25, v8, v25, v0.t
; CHECK-NEXT:    vfmv.f.s fa0, v25
; CHECK-NEXT:    ret
  %r = call reassoc double @llvm.vp.reduce.fadd.nxv1f64(double %s, <vscale x 1 x double> %v, <vscale x 1 x i1> %m, i32 %evl)
  ret double %r
}

define double @vpreduce_ord_fadd_nxv1f64(double %s, <vscale x 1 x double> %v, <vscale x 1 x i1> %m, i32 zeroext %evl) {
; CHECK-LABEL: vpreduce_ord_fadd_nxv1f64:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetvli a1, zero, e64, m1, ta, mu
; CHECK-NEXT:    vfmv.v.f v25, fa0
; CHECK-NEXT:    vsetvli zero, a0, e64, m1, tu, mu
; CHECK-NEXT:    vfredosum.vs v25, v8, v25, v0.t
; CHECK-NEXT:    vfmv.f.s fa0, v25
; CHECK-NEXT:    ret
  %r = call double @llvm.vp.reduce.fadd.nxv1f64(double %s, <vscale x 1 x double> %v, <vscale x 1 x i1> %m, i32 %evl)
  ret double %r
}

declare double @llvm.vp.reduce.fadd.nxv2f64(double, <vscale x 2 x double>, <vscale x 2 x i1>, i32)

define double @vpreduce_fadd_nxv2f64(double %s, <vscale x 2 x double> %v, <vscale x 2 x i1> %m, i32 zeroext %evl) {
; CHECK-LABEL: vpreduce_fadd_nxv2f64:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetvli a1, zero, e64, m1, ta, mu
; CHECK-NEXT:    vfmv.v.f v25, fa0
; CHECK-NEXT:    vsetvli zero, a0, e64, m2, tu, mu
; CHECK-NEXT:    vfredsum.vs v25, v8, v25, v0.t
; CHECK-NEXT:    vfmv.f.s fa0, v25
; CHECK-NEXT:    ret
  %r = call reassoc double @llvm.vp.reduce.fadd.nxv2f64(double %s, <vscale x 2 x double> %v, <vscale x 2 x i1> %m, i32 %evl)
  ret double %r
}

define double @vpreduce_ord_fadd_nxv2f64(double %s, <vscale x 2 x double> %v, <vscale x 2 x i1> %m, i32 zeroext %evl) {
; CHECK-LABEL: vpreduce_ord_fadd_nxv2f64:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetvli a1, zero, e64, m1, ta, mu
; CHECK-NEXT:    vfmv.v.f v25, fa0
; CHECK-NEXT:    vsetvli zero, a0, e64, m2, tu, mu
; CHECK-NEXT:    vfredosum.vs v25, v8, v25, v0.t
; CHECK-NEXT:    vfmv.f.s fa0, v25
; CHECK-NEXT:    ret
  %r = call double @llvm.vp.reduce.fadd.nxv2f64(double %s, <vscale x 2 x double> %v, <vscale x 2 x i1> %m, i32 %evl)
  ret double %r
}

declare double @llvm.vp.reduce.fadd.nxv4f64(double, <vscale x 4 x double>, <vscale x 4 x i1>, i32)

define double @vpreduce_fadd_nxv4f64(double %s, <vscale x 4 x double> %v, <vscale x 4 x i1> %m, i32 zeroext %evl) {
; CHECK-LABEL: vpreduce_fadd_nxv4f64:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetvli a1, zero, e64, m1, ta, mu
; CHECK-NEXT:    vfmv.v.f v25, fa0
; CHECK-NEXT:    vsetvli zero, a0, e64, m4, tu, mu
; CHECK-NEXT:    vfredsum.vs v25, v8, v25, v0.t
; CHECK-NEXT:    vfmv.f.s fa0, v25
; CHECK-NEXT:    ret
  %r = call reassoc double @llvm.vp.reduce.fadd.nxv4f64(double %s, <vscale x 4 x double> %v, <vscale x 4 x i1> %m, i32 %evl)
  ret double %r
}

define double @vpreduce_ord_fadd_nxv4f64(double %s, <vscale x 4 x double> %v, <vscale x 4 x i1> %m, i32 zeroext %evl) {
; CHECK-LABEL: vpreduce_ord_fadd_nxv4f64:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetvli a1, zero, e64, m1, ta, mu
; CHECK-NEXT:    vfmv.v.f v25, fa0
; CHECK-NEXT:    vsetvli zero, a0, e64, m4, tu, mu
; CHECK-NEXT:    vfredosum.vs v25, v8, v25, v0.t
; CHECK-NEXT:    vfmv.f.s fa0, v25
; CHECK-NEXT:    ret
  %r = call double @llvm.vp.reduce.fadd.nxv4f64(double %s, <vscale x 4 x double> %v, <vscale x 4 x i1> %m, i32 %evl)
  ret double %r
}
